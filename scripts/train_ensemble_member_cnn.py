import argparse
import json
import shutil
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from lesionshiftai.core.config import load_config
from lesionshiftai.core.distributed import barrier, cleanup_dist, setup_dist
from lesionshiftai.core.reproducibility import set_seed
from lesionshiftai.core.runtime import write_json
from lesionshiftai.data.datamodule import build_isic_fold_data_bundle
from lesionshiftai.data.split import summarize_fold_assignment
from lesionshiftai.eval.evaluator import evaluate_loader
from lesionshiftai.eval.metrics import compute_binary_metrics
from lesionshiftai.models.cnn import BaselineCNN
from lesionshiftai.train.engine import train_one_epoch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _pos_weight(train_df) -> torch.Tensor:
    counts = train_df["label"].value_counts().to_dict()
    neg = float(counts.get(0, 0))
    pos = float(counts.get(1, 1))
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


def _ensemble_root(
    output_root: Path,
    experiment_name: str,
    ensemble_run_id: str
) -> Path:
    return output_root / experiment_name / f"ensemble_{ensemble_run_id}"


def _member_dir_from_root(ensemble_root: Path, fold_index: int) -> Path:
    return (
        ensemble_root
        / "members"
        / f"fold_{fold_index}"
    )


def _prepare_member_dirs(member_dir: Path, config_path: str | Path) -> None:
    (member_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (member_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (member_dir / "predictions").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, member_dir / "config.yml")


def _write_ensemble_validation_if_ready(
    ensemble_root: Path,
    ensemble_run_id: str,
    num_folds: int,
    threshold: float
) -> dict[str, object]:
    missing_folds = []
    for fold_index in range(num_folds):
        complete_path = (
            _member_dir_from_root(ensemble_root, fold_index)
            / "metrics"
            / "member_complete.json"
        )
        if not complete_path.exists():
            missing_folds.append(fold_index)

    if missing_folds:
        return {
            "status": "pending",
            "reason": "waiting_for_remaining_folds",
            "missing_folds": missing_folds
        }

    all_val_preds = []
    member_summary_rows = []

    for fold_index in range(num_folds):
        member_dir = _member_dir_from_root(ensemble_root, fold_index)
        preds_path = member_dir / "predictions" / "val_final.csv"
        metrics_path = member_dir / "metrics" / "val_metrics.json"

        val_preds = pd.read_csv(preds_path).copy()
        required_cols = {
            "sample_id",
            "dataset",
            "label",
            "prob_malignant",
            "pred_label"
        }
        missing_cols = required_cols.difference(val_preds.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in {preds_path}: {sorted(missing_cols)}"
            )

        val_preds["member_fold"] = fold_index
        all_val_preds.append(val_preds)

        member_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        member_summary = {
            "member_fold": int(fold_index),
            "n_val_samples": int(len(val_preds))
        }
        member_summary.update(member_metrics)
        member_summary_rows.append(member_summary)

    all_preds_df = pd.concat(all_val_preds, axis=0, ignore_index=True)
    duplicate_rows = all_preds_df.duplicated(subset=["dataset", "sample_id"])
    if duplicate_rows.any():
        duplicate_count = int(duplicate_rows.sum())
        raise RuntimeError(
            "Found duplicate sample predictions while aggregating ISIC val outputs: "
            f"{duplicate_count} duplicate rows"
        )

    y_true = all_preds_df["label"].to_numpy(dtype=int)
    y_prob = all_preds_df["prob_malignant"].to_numpy(dtype=float)
    aggregate_metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold
    )
    aggregate_metrics["num_folds"] = int(num_folds)
    aggregate_metrics["threshold"] = float(threshold)
    aggregate_metrics["n_samples"] = int(len(all_preds_df))
    aggregate_metrics["ensemble_run_id"] = ensemble_run_id

    ensemble_out_dir = ensemble_root / "ensemble"
    predictions_dir = ensemble_out_dir / "predictions"
    metrics_dir = ensemble_out_dir / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    all_preds_df.to_csv(
        predictions_dir / "isic_val_aggregate_predictions.csv",
        index=False
    )
    pd.DataFrame(member_summary_rows).sort_values(
        by="member_fold"
    ).to_csv(metrics_dir / "member_val_metrics.csv", index=False)
    write_json(metrics_dir / "isic_val_aggregate_metrics.json",
               aggregate_metrics)
    write_json(
        metrics_dir / "aggregate_summary.json",
        {
            "ensemble_run_id": ensemble_run_id,
            "num_folds": int(num_folds),
            "n_aggregate_rows": int(len(all_preds_df)),
            "member_folds": list(range(num_folds))
        }
    )

    return {
        "status": "completed",
        "reason": "all_fold_members_complete",
        "missing_folds": []
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/baseline_cnn.yml", type=str)
    p.add_argument("--num-folds", default=5, type=int)
    p.add_argument(
        "--fold-index",
        default=None,
        type=int,
        help="Optional: train only one fold. Default trains all folds."
    )
    p.add_argument("--ensemble-run-id", required=True, type=str)
    p.add_argument("--threshold", default=0.5, type=float)
    args = p.parse_args()

    if args.num_folds < 2:
        raise ValueError("`--num-folds` must be >= 2")

    cfg = load_config(args.config)
    if cfg.train.epochs < 1:
        raise ValueError("`train.epochs` must be >= 1")

    dist_state = setup_dist()
    try:
        if args.fold_index is None:
            fold_indices = list(range(args.num_folds))
        else:
            if args.fold_index < 0 or args.fold_index >= args.num_folds:
                raise ValueError("`--fold-index` must be in [0, --num-folds)")
            fold_indices = [args.fold_index]

        ensemble_root = _ensemble_root(
            output_root=cfg.output_root,
            experiment_name=cfg.name,
            ensemble_run_id=args.ensemble_run_id
        )

        for fold_index in fold_indices:
            set_seed(
                cfg.seed + fold_index + dist_state.rank,
                cfg.deterministic
            )

            member_dir = _member_dir_from_root(ensemble_root, fold_index)
            if dist_state.is_main:
                _prepare_member_dirs(member_dir, args.config)
            if dist_state.enabled:
                barrier()

            bundle = build_isic_fold_data_bundle(
                cfg=cfg,
                num_folds=args.num_folds,
                fold_index=fold_index,
                world_size=dist_state.world_size,
                rank=dist_state.rank
            )
            device = dist_state.device

            model = BaselineCNN(pretrained=True).to(device)
            if dist_state.enabled:
                if device.type == "cuda":
                    model = DDP(
                        model,
                        device_ids=[dist_state.local_rank],
                        output_device=dist_state.local_rank
                    )
                else:
                    model = DDP(model)

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=_pos_weight(bundle.train_df).to(device)
            )
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.train.lr,
                weight_decay=cfg.train.weight_decay
            )

            best_pr_auc = float("-inf")
            best_epoch = -1
            best_val_metrics = None
            best_val_preds = None
            history = []

            for epoch in range(1, cfg.train.epochs + 1):
                if bundle.train_sampler is not None:
                    bundle.train_sampler.set_epoch(epoch)

                train_metrics = train_one_epoch(
                    model=model,
                    loader=bundle.train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    dist_state=dist_state,
                    threshold=args.threshold
                )
                val_metrics, val_preds = evaluate_loader(
                    model=model,
                    loader=bundle.val_loader,
                    criterion=criterion,
                    device=device,
                    dist_state=dist_state,
                    threshold=args.threshold
                )

                if dist_state.is_main:
                    history.append(
                        {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                    )
                    if epoch == 1 or val_metrics["pr_auc"] > best_pr_auc:
                        best_pr_auc = val_metrics["pr_auc"]
                        best_epoch = epoch
                        best_val_metrics = dict(val_metrics)
                        best_val_preds = val_preds.copy()
                        torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": _unwrap_model(model).state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "val_metrics": val_metrics
                            },
                            member_dir / "checkpoints" / "best.pt"
                        )
                        val_preds.to_csv(
                            member_dir / "predictions" / "val_best.csv",
                            index=False
                        )

            if dist_state.enabled:
                barrier()

            if dist_state.is_main:
                if best_val_metrics is None or best_val_preds is None:
                    raise RuntimeError(
                        "No best validation snapshot was captured during training"
                    )

                fold_summary = summarize_fold_assignment(
                    fold_df=bundle.fold_assignment_df,
                    num_folds=args.num_folds
                )
                split_summary = {
                    "fold_index": fold_index,
                    "num_folds": args.num_folds,
                    "n_fold_samples": int(
                        (bundle.fold_assignment_df["fold"] == fold_index).sum()
                    ),
                    "n_train": int(len(bundle.train_df)),
                    "n_val": int(len(bundle.val_df)),
                    "label_counts_train": {
                        str(k): int(v)
                        for k, v in (
                            bundle.train_df["label"].value_counts(
                            ).to_dict().items()
                        )
                    },
                    "label_counts_val": {
                        str(k): int(v)
                        for k, v in (
                            bundle.val_df["label"].value_counts(
                            ).to_dict().items()
                        )
                    }
                }

                # Keep existing artifact contract while avoiding a second val pass.
                best_val_preds.to_csv(
                    member_dir / "predictions" / "val_final.csv",
                    index=False
                )
                write_json(member_dir / "metrics" /
                           "history.json", {"epochs": history})
                write_json(
                    member_dir / "metrics" / "val_metrics.json",
                    best_val_metrics
                )
                write_json(
                    member_dir / "metrics" / "split_summary.json",
                    split_summary
                )
                write_json(
                    member_dir / "metrics" / "fold_assignment_summary.json",
                    fold_summary
                )
                write_json(
                    member_dir / "metrics" / "best_epoch.json",
                    {"best_epoch": int(best_epoch)}
                )
                write_json(
                    member_dir / "metrics" / "member_complete.json",
                    {
                        "fold_index": int(fold_index),
                        "status": "complete"
                    }
                )

                ensemble_aggregation = _write_ensemble_validation_if_ready(
                    ensemble_root=ensemble_root,
                    ensemble_run_id=args.ensemble_run_id,
                    num_folds=args.num_folds,
                    threshold=args.threshold
                )
                write_json(
                    member_dir / "metrics" / "ensemble_aggregation_status.json",
                    ensemble_aggregation
                )

            if dist_state.enabled:
                barrier()
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
