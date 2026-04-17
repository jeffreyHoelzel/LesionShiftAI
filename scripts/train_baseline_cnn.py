import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from lesionshiftai.core.config import load_config
from lesionshiftai.core.distributed import barrier, cleanup_dist, setup_dist
from lesionshiftai.core.reproducibility import set_seed
from lesionshiftai.core.runtime import create_run_dir, write_json
from lesionshiftai.data.datamodule import build_data_bundle
from lesionshiftai.eval.evaluator import evaluate_loader, generalization_gap
from lesionshiftai.models.cnn import BaselineCNN
from lesionshiftai.train.engine import train_one_epoch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _pos_weight(train_df) -> torch.Tensor:
    counts = train_df["label"].value_counts().to_dict()
    neg = float(counts.get(0, 0))
    pos = float(counts.get(1, 1))
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/baseline_cnn.yml", type=str)
    p.add_argument("--threshold", default=0.5, type=float)
    args = p.parse_args()

    cfg = load_config(args.config)
    process_rank = int(os.environ.get("RANK", "0"))
    set_seed(cfg.seed + process_rank, cfg.deterministic)

    dist_state = setup_dist()
    try:
        run_dir = create_run_dir(
            cfg, args.config) if dist_state.is_main else None
        run_dir_box = [str(run_dir) if run_dir is not None else ""]
        if dist_state.enabled:
            dist.broadcast_object_list(run_dir_box, src=0)
        run_dir = Path(run_dir_box[0])

        bundle = build_data_bundle(
            cfg,
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
            model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )

        best_pr_auc = float("-inf")
        history = []

        for epoch in range(1, cfg.train.epochs + 1):
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

            if dist_state.is_main:
                history.append(
                    {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                )

                if epoch == 1 or val_metrics["pr_auc"] > best_pr_auc:
                    best_pr_auc = val_metrics["pr_auc"]
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": _unwrap_model(model).state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_metrics": val_metrics
                        },
                        run_dir / "checkpoints" / "best.pt"
                    )
                    val_preds.to_csv(
                        run_dir / "predictions" / "val_best.csv", index=False
                    )

        if dist_state.enabled:
            barrier()

        ckpt = torch.load(run_dir / "checkpoints" /
                          "best.pt", map_location=device)
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
            val_preds.to_csv(run_dir / "predictions" /
                             "val_final.csv", index=False)
            test_preds.to_csv(run_dir / "predictions" /
                              "ham_test.csv", index=False)

            write_json(run_dir / "metrics" /
                       "history.json", {"epochs": history})
            write_json(run_dir / "metrics" / "val_metrics.json", val_metrics)
            write_json(run_dir / "metrics" / "test_metrics.json", test_metrics)
            write_json(run_dir / "metrics" / "generalization_gap.json", gap)
    finally:
        cleanup_dist()


if __name__ == "__main__":
    main()
