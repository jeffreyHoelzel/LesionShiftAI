import argparse
import torch
from torch import nn
from torch.optim import AdamW
from lesionshiftai.core.config import load_config
from lesionshiftai.core.reproducibility import set_seed
from lesionshiftai.core.runtime import create_run_dir, write_json
from lesionshiftai.data.datamodule import build_data_bundle
from lesionshiftai.eval.evaluator import evaluate_loader, generalization_gap
from lesionshiftai.models.cnn import BaselineCNN
from lesionshiftai.train.engine import train_one_epoch


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    set_seed(cfg.seed, cfg.deterministic)
    run_dir = create_run_dir(cfg, args.config)

    bundle = build_data_bundle(cfg)
    device = _device()

    model = BaselineCNN(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=_pos_weight(bundle.train_df).to(device))
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr,
                      weight_decay=cfg.train.weight_decay)

    best_pr_auc = -1.0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            model, bundle.train_loader, optimizer, criterion, device, args.threshold
        )
        val_metrics, val_preds = evaluate_loader(
            model, bundle.val_loader, criterion, device, args.threshold
        )

        history.append(
            {"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics
                },
                run_dir / "checkpoints" / "best.pt"
            )
            val_preds.to_csv(run_dir / "predictions" /
                             "val_best.csv", index=False)

    ckpt = torch.load(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_preds = evaluate_loader(
        model, bundle.val_loader, criterion, device, args.threshold
    )
    test_metrics, test_preds = evaluate_loader(
        model, bundle.test_loader, criterion, device, args.threshold
    )
    gap = generalization_gap(val_metrics, test_metrics)

    val_preds.to_csv(run_dir / "predictions" / "val_final.csv", index=False)
    test_preds.to_csv(run_dir / "predictions" / "ham_test.csv", index=False)

    write_json(run_dir / "metrics" / "history.json", {"epochs": history})
    write_json(run_dir / "metrics" / "val_metrics.json", val_metrics)
    write_json(run_dir / "metrics" / "test_metrics.json", test_metrics)
    write_json(run_dir / "metrics" / "generalization_gap.json", gap)


if __name__ == "__main__":
    main()
