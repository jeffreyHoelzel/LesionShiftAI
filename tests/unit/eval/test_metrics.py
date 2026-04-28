import numpy as np
import pytest

from lesionshiftai.eval.metrics import compute_binary_metrics


pytestmark = pytest.mark.unit


def test_compute_binary_metrics_basic() -> None:
    y_true = np.array([0, 0, 1, 1], dtype=int)
    y_prob = np.array([0.1, 0.9, 0.2, 0.8], dtype=float)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_compute_binary_metrics_single_class_edge_case() -> None:
    y_true = np.zeros(6, dtype=int)
    y_prob = np.linspace(0.1, 0.6, 6)

    with pytest.warns(Warning):
        metrics = compute_binary_metrics(y_true, y_prob)

    assert np.isnan(metrics["roc_auc"])
    assert metrics["pr_auc"] == pytest.approx(0.0)
