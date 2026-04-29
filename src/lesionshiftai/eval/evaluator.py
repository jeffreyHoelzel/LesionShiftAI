"""evaluator.py

Runs the HAM10000 external dataset testing step.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from lesionshiftai.core.distributed import DistState, all_gather_object
from lesionshiftai.eval.metrics import compute_binary_metrics


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    dist_state: Optional[DistState] = None,
    threshold: float = 0.5
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates a model on a DataLoader and returns metrics with prediction rows.

    Parameters
    ------------
        model : torch.nn.Module
            Model used to generate logits from input images.
        loader : DataLoader
            DataLoader containing evaluation batches.
        criterion : torch.nn.Module
            Loss function used to calculate evaluation loss.
        device : torch.device
            Device used for model inputs and labels.
        dist_state : Optional[DistState]
            Optional distributed state used to gather predictions across processes.
        threshold : float
            Probability cutoff used to convert predicted probabilities into binary labels.

    Returns
    --------
        result : Tuple[Dict[str, Any], pd.DataFrame]
            Evaluation metrics dictionary and prediction DataFrame.

    Raises
    -------
        KeyError
            Raised when evaluation batches are missing required fields.
        RuntimeError
            Raised when model evaluation, distributed gathering, or metric computation fails.
    """
    model.eval()

    y_true = []
    y_prob = []
    loss_sum = 0.0
    n = 0
    sample_ids = []
    datasets = []

    # iterate over batches and calculate loss, populate lists
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
        sample_ids.extend(batch["sample_id"])
        datasets.extend(batch["dataset"])

        batch_n = int(labels.numel())
        loss_sum += float(loss.item()) * batch_n
        n += batch_n

    # aggregate evaluation metrics across ranks
    payload = {
        "y_true": y_true,
        "y_prob": y_prob,
        "loss_sum": loss_sum,
        "n": n,
        "sample_id": sample_ids,
        "dataset": datasets
    }
    gathered = all_gather_object(payload) if (
        dist_state and dist_state.enabled) else [payload]

    y_true_all = []
    y_prob_all = []
    sample_id_all = []
    dataset_all = []
    loss_sum_all = 0.0
    n_all = 0
    for part in gathered:
        y_true_all.extend(part["y_true"])
        y_prob_all.extend(part["y_prob"])
        sample_id_all.extend(part["sample_id"])
        dataset_all.extend(part["dataset"])
        loss_sum_all += float(part["loss_sum"])
        n_all += int(part["n"])

    y_true_np = np.asarray(y_true_all, dtype=int)
    y_prob_np = np.asarray(y_prob_all, dtype=float)

    preds = pd.DataFrame(
        {
            "sample_id": sample_id_all,
            "dataset": dataset_all,
            "label": y_true_np,
            "prob_malignant": y_prob_np,
            "pred_label": (y_prob_np >= threshold).astype(int)
        }
    )

    # DistributedSampler can pad partitions, dedupe for final prediction/metric rows
    preds = preds.drop_duplicates(
        subset=["dataset", "sample_id"], keep="first"
    ).reset_index(drop=True)

    y_true_final = preds["label"].to_numpy(dtype=int)
    y_prob_final = preds["prob_malignant"].to_numpy(dtype=float)

    metrics = compute_binary_metrics(
        y_true_final, y_prob_final, threshold=threshold)
    metrics["loss"] = loss_sum_all / max(n_all, 1)

    return metrics, preds


def generalization_gap(val_metrics: Dict[str, Any], test_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes validation-minus-test gaps for core binary classification metrics.

    Parameters
    ------------
        val_metrics : Dict[str, Any]
            Validation metrics dictionary.
        test_metrics : Dict[str, Any]
            Test metrics dictionary.

    Returns
    --------
        gaps : Dict[str, Any]
            Dictionary containing validation-minus-test metric gaps.

    Raises
    -------
        TypeError
            Raised when shared metric values cannot be converted to floats or subtracted.
    """
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    return {
        f"{k}_gap_val_minus_test": float(val_metrics[k] - test_metrics[k])
        for k in keys
        if k in val_metrics and k in test_metrics
    }
