from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lesionshiftai.eval.metrics import compute_binary_metrics


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run model evaluation on a dataloader and return metrics + predictions."""
    model.eval()

    losses = []
    y_true = []
    y_prob = []
    sample_ids = []
    datasets = []

    # iterate over batches and calculate loss, populate lists
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
        sample_ids.extend(batch["sample_id"])
        datasets.extend(batch["dataset"])

    y_true_np = np.asarray(y_true, dtype=int)
    y_prob_np = np.asarray(y_prob, dtype=float)

    metrics = compute_binary_metrics(y_true_np, y_prob_np, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    preds = pd.DataFrame({
        "sample_id": sample_ids,
        "dataset": datasets,
        "label": y_true_np,
        "prob_malignant": y_prob_np,
        "pred_label": (y_prob_np >= threshold).astype(int)
    })

    return metrics, preds


def generalization_gap(val_metrics: Dict[str, Any], test_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Compute validation-minus-test gap for core performance metrics."""
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return {
        f"{k}_gap_val_minus_test": float(val_metrics[k] - test_metrics[k])
        for k in keys
        if k in val_metrics and k in test_metrics
    }
