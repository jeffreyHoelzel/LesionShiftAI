from typing import Any, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from lesionshiftai.core.distributed import DistState, all_gather_object
from lesionshiftai.eval.metrics import compute_binary_metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    dist_state: Optional[DistState] = None,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Train for one epoch and return aggregate training metrics."""
    model.train()

    y_true = []
    y_prob = []
    loss_sum = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # optimize model (backprop)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # 0 = benign, 1 = malignant
        probs = torch.sigmoid(logits)

        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

        batch_n = int(labels.numel())
        loss_sum += float(loss.item()) * batch_n
        n += batch_n

    # aggregate training metrics across ranks
    payload = {
        "y_true": y_true,
        "y_prob": y_prob,
        "loss_sum": loss_sum,
        "n": n
    }
    gathered = all_gather_object(payload) if (
        dist_state and dist_state.enabled) else [payload]

    y_true_all = []
    y_prob_all = []
    loss_sum_all = 0.0
    n_all = 0
    for p in gathered:
        y_true_all.extend(p["y_true"])
        y_prob_all.extend(p["y_prob"])
        loss_sum_all += float(p["loss_sum"])
        n_all += int(p["n"])

    y_true_np = np.asarray(y_true_all, dtype=int)
    y_prob_np = np.asarray(y_prob_all, dtype=float)

    metrics = compute_binary_metrics(y_true_np, y_prob_np, threshold=threshold)
    metrics["loss"] = loss_sum_all / max(n_all, 1)

    return metrics
