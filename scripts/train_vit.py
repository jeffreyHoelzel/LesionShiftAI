import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from lesionshiftai.core.config import ExperimentConfig, load_config
from lesionshiftai.core.distributed import barrier, cleanup_dist, setup_dist
from lesionshiftai.core.reproducibility import set_seed
from lesionshiftai.core.runtime import create_run_dir, write_json


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _pos_weight(train_df) -> torch.Tensor:
    counts = train_df["label"].value_counts().to_dict()
    neg = float(counts.get(0, 0))
    pos = float(counts.get(1, 1))
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


def _build_scheduler(optimizer: AdamW, cfg: ExperimentConfig):
    cosine_epochs = max(cfg.train.epochs - cfg.train.warmup_epochs, 1)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=cfg.train.min_lr
    )
    if cfg.train.warmup_epochs == 0:
        return cosine

    # Warm up linearly from a lower starting LR for stable ViT fine-tuning.
    warmup = LinearLR(
        optimizer,
        start_factor=1.0 / float(cfg.train.warmup_epochs + 1),
        end_factor=1.0,
        total_iters=cfg.train.warmup_epochs
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.train.warmup_epochs]
    )


def _infer_run_dir_from_checkpoint(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def _resolve_run_dir(
    cfg: ExperimentConfig,
    config_path: str,
    resume: str | None,
    is_main: bool,
    dist_enabled: bool
) -> tuple[Path, Path | None]:
    run_dir: Path | None = None
    resume_path: Path | None = None

    if is_main:
        if resume is None:
            run_dir = create_run_dir(cfg, config_path)
        else:
            resume_path = Path(resume).expanduser().resolve()
            if not resume_path.exists():
                raise FileNotFoundError(
                    f"Resume checkpoint not found: {resume_path}"
                )
            run_dir = _infer_run_dir_from_checkpoint(resume_path)
            (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
            (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
            (run_dir / "predictions").mkdir(parents=True, exist_ok=True)

    run_dir_box = [str(run_dir) if run_dir is not None else ""]
    resume_box = [str(resume_path) if resume_path is not None else ""]
    if dist_enabled:
        dist.broadcast_object_list(run_dir_box, src=0)
        dist.broadcast_object_list(resume_box, src=0)

    resolved_run_dir = Path(run_dir_box[0])
    resolved_resume = Path(resume_box[0]) if resume_box[0] else None
    return resolved_run_dir, resolved_resume


def _load_existing_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    rows = payload.get("epochs", [])
    return rows if isinstance(rows, list) else []


def _build_checkpoint_payload(
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler,
    epoch: int,
    val_metrics: dict,
    best_pr_auc: float,
    resumed_from: str | None
) -> dict:
    return {
        "epoch": int(epoch),
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_metrics": val_metrics,
        "best_pr_auc": float(best_pr_auc),
        "resumed_from": resumed_from
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/vit_b16.yml", type=str)
    p.add_argument("--threshold", default=0.5, type=float)
    p.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Optional checkpoint path to resume ViT training."
    )
    args = p.parse_args()

    from lesionshiftai.data.datamodule import build_data_bundle
    from lesionshiftai.eval.evaluator import evaluate_loader, generalization_gap
    from lesionshiftai.models.vit import ViTBinaryClassifier
    from lesionshiftai.train.engine import train_one_epoch

    cfg = load_config(args.config)
    process_rank = int(os.environ.get("RANK", "0"))
    set_seed(cfg.seed + process_rank, cfg.deterministic)

    dist_state = setup_dist()
    try:
        run_dir, resume_path = _resolve_run_dir(
            cfg=cfg,
            config_path=args.config,
            resume=args.resume,
            is_main=dist_state.is_main,
            dist_enabled=dist_state.enabled
        )

        bundle = build_data_bundle(
            cfg,
            world_size=dist_state.world_size,
            rank=dist_state.rank
        )
        device = dist_state.device

        model = ViTBinaryClassifier(pretrained=True).to(device)
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
        scheduler = _build_scheduler(optimizer, cfg)

        best_pr_auc = float("-inf")
        start_epoch = 1
        resumed_from = str(resume_path) if resume_path else None

        if resume_path is not None:
            ckpt = torch.load(resume_path, map_location=device)
            _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_pr_auc = float(
                ckpt.get(
                    "best_pr_auc",
                    ckpt.get("val_metrics", {}).get("pr_auc", float("-inf"))
                )
            )

        history = []
        if dist_state.is_main:
            history = _load_existing_history(run_dir / "metrics" / "history.json")

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            if bundle.train_sampler is not None:
                bundle.train_sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                model,
                bundle.train_loader,
                optimizer,
                criterion,
                device,
                dist_state=dist_state,
                threshold=args.threshold
            )
            val_metrics, val_preds = evaluate_loader(
                model,
                bundle.val_loader,
                criterion,
                device,
                dist_state=dist_state,
                threshold=args.threshold
            )
            scheduler.step()

            if dist_state.is_main:
                history.append(
                    {
                        "epoch": epoch,
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "train": train_metrics,
                        "val": val_metrics
                    }
                )

                ckpt_payload = _build_checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    best_pr_auc=max(best_pr_auc, float(val_metrics["pr_auc"])),
                    resumed_from=resumed_from
                )
                torch.save(ckpt_payload, run_dir / "checkpoints" / "last.pt")

                if best_pr_auc == float("-inf") or val_metrics["pr_auc"] > best_pr_auc:
                    best_pr_auc = float(val_metrics["pr_auc"])
                    ckpt_payload["best_pr_auc"] = best_pr_auc
                    torch.save(ckpt_payload, run_dir / "checkpoints" / "best.pt")
                    val_preds.to_csv(
                        run_dir / "predictions" / "val_best.csv",
                        index=False
                    )

        if dist_state.enabled:
            barrier()

        best_ckpt_path = run_dir / "checkpoints" / "best.pt"
        if not best_ckpt_path.exists():
            if resume_path is not None and resume_path.exists():
                best_ckpt_path = resume_path
            else:
                raise FileNotFoundError(
                    "No checkpoint available for final evaluation "
                    f"at {best_ckpt_path}"
                )

        ckpt = torch.load(best_ckpt_path, map_location=device)
        _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])

        val_metrics, val_preds = evaluate_loader(
            model,
            bundle.val_loader,
            criterion,
            device,
            dist_state=dist_state,
            threshold=args.threshold
        )
        test_metrics, test_preds = evaluate_loader(
            model,
            bundle.test_loader,
            criterion,
            device,
            dist_state=dist_state,
            threshold=args.threshold
        )
        gap = generalization_gap(val_metrics, test_metrics)

        if dist_state.is_main:
            val_preds.to_csv(run_dir / "predictions" / "val_final.csv", index=False)
            test_preds.to_csv(run_dir / "predictions" / "ham_test.csv", index=False)

            write_json(run_dir / "metrics" / "history.json", {"epochs": history})
            write_json(run_dir / "metrics" / "val_metrics.json", val_metrics)
            write_json(run_dir / "metrics" / "test_metrics.json", test_metrics)
            write_json(run_dir / "metrics" / "generalization_gap.json", gap)
            write_json(
                run_dir / "metrics" / "resume.json",
                {
                    "resumed": resume_path is not None,
                    "resume_checkpoint": resumed_from,
                    "start_epoch": int(start_epoch),
                    "configured_epochs": int(cfg.train.epochs)
                }
            )
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
