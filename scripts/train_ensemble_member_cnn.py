import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lesionshiftai.core.config import load_config
from lesionshiftai.core.distributed import barrier, cleanup_dist, setup_dist
from lesionshiftai.core.reproducibility import init_generator, seed_worker, set_seed
from lesionshiftai.core.runtime import write_json
from lesionshiftai.data.datamodule import build_isic_fold_data_bundle
from lesionshiftai.data.dataset import LesionDataset
from lesionshiftai.data.metadata import load_ham_metadata
from lesionshiftai.data.split import summarize_fold_assignment
from lesionshiftai.data.transforms import build_eval_transform
from lesionshiftai.eval.curves import (
    write_binary_curve_artifacts,
    write_fold_curve_overlay_artifacts,
)
from lesionshiftai.eval.evaluator import evaluate_loader, generalization_gap
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


def _build_ham_test_loader(
    cfg,
    world_size: int,
    rank: int,
) -> DataLoader:
    ham_df = load_ham_metadata(cfg.data.ham_root)
    eval_tf = build_eval_transform(cfg.data.image_size)
    test_ds = LesionDataset(ham_df, eval_tf)

    test_sampler = None
    if world_size > 1:
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=cfg.seed,
        )

    return DataLoader(
        test_ds,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.data.num_workers > 0,
        shuffle=False,
        sampler=test_sampler,
        generator=init_generator(cfg.seed + 9000 + rank),
    )


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
    member_curve_payloads = []
    all_test_preds = []
    member_test_summary_rows = []
    member_test_curve_payloads = []
    required_cols = {"sample_id", "dataset", "label", "prob_malignant", "pred_label"}

    for fold_index in range(num_folds):
        member_dir = _member_dir_from_root(ensemble_root, fold_index)
        preds_path = member_dir / "predictions" / "val_final.csv"
        metrics_path = member_dir / "metrics" / "val_metrics.json"
        curve_path = member_dir / "metrics" / "curves" / "val_final_curves.json"
        test_preds_path = member_dir / "predictions" / "ham_test.csv"
        test_metrics_path = member_dir / "metrics" / "test_metrics.json"
        test_curve_path = member_dir / "metrics" / "curves" / "ham_test_curves.json"

        val_preds = pd.read_csv(preds_path).copy()
        missing_cols = required_cols.difference(val_preds.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in {preds_path}: {sorted(missing_cols)}"
            )
        val_duplicates = val_preds.duplicated(subset=["dataset", "sample_id"])
        if val_duplicates.any():
            duplicate_count = int(val_duplicates.sum())
            raise RuntimeError(
                f"Found duplicate member val predictions in {preds_path}: "
                f"{duplicate_count} rows"
            )

        val_preds["member_fold"] = fold_index
        all_val_preds.append(val_preds)

        member_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        member_curve_payloads.append(
            json.loads(curve_path.read_text(encoding="utf-8"))
        )
        member_summary = {
            "member_fold": int(fold_index),
            "n_val_samples": int(len(val_preds))
        }
        member_summary.update(member_metrics)
        member_summary_rows.append(member_summary)

        test_preds = pd.read_csv(test_preds_path).copy()
        missing_test_cols = required_cols.difference(test_preds.columns)
        if missing_test_cols:
            raise ValueError(
                f"Missing columns in {test_preds_path}: {sorted(missing_test_cols)}"
            )
        test_duplicates = test_preds.duplicated(subset=["dataset", "sample_id"])
        if test_duplicates.any():
            duplicate_count = int(test_duplicates.sum())
            raise RuntimeError(
                f"Found duplicate member HAM predictions in {test_preds_path}: "
                f"{duplicate_count} rows"
            )

        test_preds["member_fold"] = fold_index
        all_test_preds.append(test_preds)

        member_test_metrics = json.loads(test_metrics_path.read_text(encoding="utf-8"))
        member_test_curve_payloads.append(
            json.loads(test_curve_path.read_text(encoding="utf-8"))
        )
        member_test_summary = {
            "member_fold": int(fold_index),
            "n_test_samples": int(len(test_preds))
        }
        member_test_summary.update(member_test_metrics)
        member_test_summary_rows.append(member_test_summary)

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
    write_binary_curve_artifacts(
        y_true=y_true,
        y_prob=y_prob,
        output_dir=metrics_dir / "curves",
        split_name="isic_val_aggregate",
        model_scope="ensemble_aggregate",
        extra_metadata={
            "threshold": float(threshold),
            "ensemble_run_id": ensemble_run_id,
            "num_folds": int(num_folds),
        },
    )
    write_fold_curve_overlay_artifacts(
        fold_curve_payloads=member_curve_payloads,
        output_dir=metrics_dir / "curves",
        split_name="isic_val_member_folds",
        model_scope="ensemble_member_folds",
        extra_metadata={
            "ensemble_run_id": ensemble_run_id,
            "num_folds": int(num_folds),
        },
    )
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

    all_test_preds_df = pd.concat(all_test_preds, axis=0, ignore_index=True)
    duplicate_test_rows = all_test_preds_df.duplicated(
        subset=["member_fold", "dataset", "sample_id"]
    )
    if duplicate_test_rows.any():
        duplicate_count = int(duplicate_test_rows.sum())
        raise RuntimeError(
            "Found duplicate member HAM predictions while aggregating "
            "external outputs: "
            f"{duplicate_count} duplicate rows"
        )

    label_consistency = (
        all_test_preds_df
        .groupby(["dataset", "sample_id"])["label"]
        .nunique()
    )
    if (label_consistency > 1).any():
        inconsistent = int((label_consistency > 1).sum())
        raise RuntimeError(
            "Found inconsistent HAM labels across members while aggregating "
            f"external outputs: {inconsistent} sample(s)"
        )

    member_counts = (
        all_test_preds_df
        .groupby(["dataset", "sample_id"])["member_fold"]
        .nunique()
    )
    missing_member_predictions = member_counts[member_counts != num_folds]
    if not missing_member_predictions.empty:
        raise RuntimeError(
            "HAM aggregate predictions are incomplete: expected all folds per sample, "
            f"but {len(missing_member_predictions)} sample(s) do not have {num_folds} "
            "member predictions"
        )

    test_aggregate_df = (
        all_test_preds_df
        .groupby(["dataset", "sample_id"], as_index=False)
        .agg(
            label=("label", "first"),
            prob_malignant=("prob_malignant", "mean"),
            prob_malignant_std=("prob_malignant", "std"),
            member_predictions=("member_fold", "nunique"),
        )
    )
    test_aggregate_df["prob_malignant_std"] = (
        test_aggregate_df["prob_malignant_std"].fillna(0.0)
    )
    test_aggregate_df["pred_label"] = (
        test_aggregate_df["prob_malignant"] >= threshold
    ).astype(int)

    test_y_true = test_aggregate_df["label"].to_numpy(dtype=int)
    test_y_prob = test_aggregate_df["prob_malignant"].to_numpy(dtype=float)
    test_aggregate_metrics = compute_binary_metrics(
        y_true=test_y_true,
        y_prob=test_y_prob,
        threshold=threshold,
    )
    test_aggregate_metrics["num_folds"] = int(num_folds)
    test_aggregate_metrics["threshold"] = float(threshold)
    test_aggregate_metrics["n_samples"] = int(len(test_aggregate_df))
    test_aggregate_metrics["ensemble_run_id"] = ensemble_run_id
    test_aggregate_metrics["aggregation"] = "mean_prob_malignant"

    test_aggregate_df.to_csv(
        predictions_dir / "ham_test_aggregate_predictions.csv",
        index=False,
    )
    pd.DataFrame(member_test_summary_rows).sort_values(
        by="member_fold"
    ).to_csv(metrics_dir / "member_test_metrics.csv", index=False)
    write_binary_curve_artifacts(
        y_true=test_y_true,
        y_prob=test_y_prob,
        output_dir=metrics_dir / "curves",
        split_name="ham_test_aggregate",
        model_scope="ensemble_aggregate",
        extra_metadata={
            "threshold": float(threshold),
            "ensemble_run_id": ensemble_run_id,
            "num_folds": int(num_folds),
        },
    )
    write_fold_curve_overlay_artifacts(
        fold_curve_payloads=member_test_curve_payloads,
        output_dir=metrics_dir / "curves",
        split_name="ham_test_member_folds",
        model_scope="ensemble_member_folds",
        extra_metadata={
            "ensemble_run_id": ensemble_run_id,
            "num_folds": int(num_folds),
        },
    )
    write_json(metrics_dir / "ham_test_aggregate_metrics.json", test_aggregate_metrics)
    write_json(
        metrics_dir / "generalization_gap.json",
        generalization_gap(aggregate_metrics, test_aggregate_metrics),
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
        ham_test_loader = _build_ham_test_loader(
            cfg=cfg,
            world_size=dist_state.world_size,
            rank=dist_state.rank,
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

            best_ckpt = torch.load(
                member_dir / "checkpoints" / "best.pt",
                map_location=device,
            )
            _unwrap_model(model).load_state_dict(best_ckpt["model_state_dict"])
            test_metrics, test_preds = evaluate_loader(
                model=model,
                loader=ham_test_loader,
                criterion=criterion,
                device=device,
                dist_state=dist_state,
                threshold=args.threshold,
            )

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
                write_binary_curve_artifacts(
                    y_true=best_val_preds["label"].to_numpy(dtype=int),
                    y_prob=best_val_preds["prob_malignant"].to_numpy(dtype=float),
                    output_dir=member_dir / "metrics" / "curves",
                    split_name="val_final",
                    model_scope="ensemble_member",
                    extra_metadata={
                        "threshold": float(args.threshold),
                        "ensemble_run_id": args.ensemble_run_id,
                        "fold_index": int(fold_index),
                        "num_folds": int(args.num_folds),
                    },
                )
                test_preds.to_csv(
                    member_dir / "predictions" / "ham_test.csv",
                    index=False
                )
                write_binary_curve_artifacts(
                    y_true=test_preds["label"].to_numpy(dtype=int),
                    y_prob=test_preds["prob_malignant"].to_numpy(dtype=float),
                    output_dir=member_dir / "metrics" / "curves",
                    split_name="ham_test",
                    model_scope="ensemble_member",
                    extra_metadata={
                        "threshold": float(args.threshold),
                        "ensemble_run_id": args.ensemble_run_id,
                        "fold_index": int(fold_index),
                        "num_folds": int(args.num_folds),
                    },
                )
                write_json(member_dir / "metrics" /
                           "history.json", {"epochs": history})
                write_json(
                    member_dir / "metrics" / "val_metrics.json",
                    best_val_metrics
                )
                write_json(
                    member_dir / "metrics" / "test_metrics.json",
                    test_metrics
                )
                write_json(
                    member_dir / "metrics" / "generalization_gap.json",
                    generalization_gap(best_val_metrics, test_metrics)
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
