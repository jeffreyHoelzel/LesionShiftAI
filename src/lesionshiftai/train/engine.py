from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from lesionshiftai.eval.metrics import compute_binary_metrics


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Train for one epoch and return aggregate training metrics."""
    model.train()

    losses = []
    y_true = []
    y_prob = []

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

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=int)
    y_prob_np = np.asarray(y_prob, dtype=float)

    metrics = compute_binary_metrics(y_true_np, y_prob_np, threshold=threshold)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    return metrics
