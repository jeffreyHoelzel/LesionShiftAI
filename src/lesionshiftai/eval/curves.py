"""curves.py

Generates and writes all curves to disk.
"""
import json
from pathlib import Path
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
matplotlib.use("Agg")


def write_binary_curve_artifacts(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str | Path,
    split_name: str,
    model_scope: str,
    extra_metadata: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Writes ROC and precision-recall curve artifacts for binary classification.

    Parameters
    ------------
        y_true : np.ndarray
            Ground truth binary labels.
        y_prob : np.ndarray
            Predicted positive class probabilities.
        output_dir : str | Path
            Directory where curve plots and JSON payloads are written.
        split_name : str
            Name of the data split being evaluated.
        model_scope : str
            Label describing the model or evaluation scope.
        extra_metadata : Dict[str, Any] | None
            Optional metadata added to the serialized curve payload.

    Returns
    --------
        payload : Dict[str, Any]
            Serialized curve payload containing metrics, curve values, and metadata.

    Raises
    -------
        OSError
            Raised when the output directory or artifact files cannot be created or written.
        ValueError
            Raised when curve metrics cannot be computed from the provided labels or probabilities.
    """
    y_true_np = np.asarray(y_true, dtype=int)
    y_prob_np = np.asarray(y_prob, dtype=float)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_samples = int(y_true_np.size)
    n_positive = int((y_true_np == 1).sum())
    n_negative = int((y_true_np == 0).sum())

    payload: Dict[str, Any] = {
        "status": "ok",
        "split": split_name,
        "model_scope": model_scope,
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative
    }
    if extra_metadata:
        payload.update(extra_metadata)

    curves_json_path = output_path / f"{split_name}_curves.json"
    if n_positive == 0 or n_negative == 0:
        payload["status"] = "skipped_single_class"
        payload["reason"] = "curve generation requires both classes"
        curves_json_path.write_text(
            _to_json(payload),
            encoding="utf-8"
        )
        return payload

    fpr, tpr, roc_thresholds = roc_curve(y_true_np, y_prob_np)
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true_np, y_prob_np)

    roc_auc = float(roc_auc_score(y_true_np, y_prob_np))
    pr_auc = float(average_precision_score(y_true_np, y_prob_np))

    payload["roc_auc"] = roc_auc
    payload["pr_auc"] = pr_auc
    payload["roc_curve"] = {
        "fpr": _to_json_numbers(fpr),
        "tpr": _to_json_numbers(tpr),
        "thresholds": _to_json_numbers(roc_thresholds)
    }
    payload["pr_curve"] = {
        "precision": _to_json_numbers(precision),
        "recall": _to_json_numbers(recall),
        "thresholds": _to_json_numbers(pr_thresholds)
    }

    _plot_roc_curve(
        output_path=output_path / f"{split_name}_roc.png",
        split_name=split_name,
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc
    )
    _plot_pr_curve(
        output_path=output_path / f"{split_name}_pr.png",
        split_name=split_name,
        precision=precision,
        recall=recall,
        pr_auc=pr_auc,
        n_positive=n_positive,
        n_samples=n_samples
    )

    curves_json_path.write_text(
        _to_json(payload),
        encoding="utf-8"
    )
    return payload


def write_fold_curve_overlay_artifacts(
    fold_curve_payloads: List[Dict[str, Any]],
    output_dir: str | Path,
    split_name: str,
    model_scope: str,
    extra_metadata: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Writes fold-level ROC and precision-recall overlay artifacts.

    Parameters
    ------------
        fold_curve_payloads : List[Dict[str, Any]]
            List of per-fold curve payloads generated from binary curve artifacts.
        output_dir : str | Path
            Directory where overlay plots and JSON payloads are written.
        split_name : str
            Name of the data split being evaluated.
        model_scope : str
            Label describing the model or evaluation scope.
        extra_metadata : Dict[str, Any] | None
            Optional metadata added to the serialized overlay payload.

    Returns
    --------
        payload : Dict[str, Any]
            Serialized overlay payload containing plotted fold summaries and skipped fold details.

    Raises
    -------
        OSError
            Raised when the output directory or artifact files cannot be created or written.
        TypeError
            Raised when fold payload values cannot be converted to expected numeric types.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "status": "ok",
        "split": split_name,
        "model_scope": model_scope,
        "n_folds_requested": int(len(fold_curve_payloads))
    }
    if extra_metadata:
        payload.update(extra_metadata)

    fold_curves: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []

    for idx, fold_payload in enumerate(fold_curve_payloads):
        fold_index_raw = fold_payload.get("fold_index")
        fold_index = int(fold_index_raw) if fold_index_raw is not None else idx
        status = str(fold_payload.get("status", "unknown"))
        if status != "ok":
            skipped_folds.append(
                {"fold_index": fold_index, "reason": f"status_{status}"}
            )
            continue

        roc_curve = fold_payload.get("roc_curve")
        pr_curve = fold_payload.get("pr_curve")
        if not isinstance(roc_curve, dict) or not isinstance(pr_curve, dict):
            skipped_folds.append(
                {"fold_index": fold_index, "reason": "missing_curve_payload"}
            )
            continue

        fpr = _to_float_array(roc_curve.get("fpr"))
        tpr = _to_float_array(roc_curve.get("tpr"))
        recall = _to_float_array(pr_curve.get("recall"))
        precision = _to_float_array(pr_curve.get("precision"))

        if fpr.size == 0 or tpr.size == 0 or recall.size == 0 or precision.size == 0:
            skipped_folds.append(
                {"fold_index": fold_index, "reason": "empty_curve_arrays"}
            )
            continue
        if fpr.size != tpr.size or recall.size != precision.size:
            skipped_folds.append(
                {"fold_index": fold_index, "reason": "mismatched_curve_lengths"}
            )
            continue

        fold_curves.append(
            {
                "fold_index": int(fold_index),
                "roc_auc": float(fold_payload.get("roc_auc", float("nan"))),
                "pr_auc": float(fold_payload.get("pr_auc", float("nan"))),
                "n_samples": int(fold_payload.get("n_samples", 0)),
                "n_positive": int(fold_payload.get("n_positive", 0)),
                "fpr": fpr,
                "tpr": tpr,
                "recall": recall,
                "precision": precision
            }
        )

    fold_curves.sort(key=lambda row: row["fold_index"])
    payload["n_folds_plotted"] = int(len(fold_curves))
    payload["skipped_folds"] = skipped_folds
    payload["folds"] = [
        {
            "fold_index": int(row["fold_index"]),
            "roc_auc": float(row["roc_auc"]),
            "pr_auc": float(row["pr_auc"]),
            "n_samples": int(row["n_samples"]),
            "n_positive": int(row["n_positive"])
        }
        for row in fold_curves
    ]

    overlay_json_path = output_path / f"{split_name}_curves.json"
    if not fold_curves:
        payload["status"] = "skipped_no_valid_folds"
        payload["reason"] = "no_valid_fold_curve_payloads"
        overlay_json_path.write_text(_to_json(payload), encoding="utf-8")
        return payload

    n_samples_total = int(sum(int(row["n_samples"]) for row in fold_curves))
    n_positive_total = int(sum(int(row["n_positive"]) for row in fold_curves))
    prevalence = (
        (n_positive_total / n_samples_total)
        if n_samples_total > 0
        else None
    )

    _plot_fold_roc_curves(
        output_path=output_path / f"{split_name}_roc.png",
        split_name=split_name,
        fold_curves=fold_curves,
    )
    _plot_fold_pr_curves(
        output_path=output_path / f"{split_name}_pr.png",
        split_name=split_name,
        fold_curves=fold_curves,
        prevalence=prevalence,
    )

    payload["n_samples"] = n_samples_total
    payload["n_positive"] = n_positive_total
    payload["n_negative"] = int(n_samples_total - n_positive_total)
    if prevalence is not None:
        payload["prevalence"] = float(prevalence)

    overlay_json_path.write_text(_to_json(payload), encoding="utf-8")
    return payload


def write_fold_auc_history_overlay_artifacts(
    fold_history_payloads: List[Dict[str, Any]],
    output_dir: str | Path,
    split_name: str,
    model_scope: str,
    extra_metadata: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Writes fold-level ROC AUC and PR AUC history overlay artifacts.

    Parameters
    ------------
        fold_history_payloads : List[Dict[str, Any]]
            List of per-fold training history payloads containing epoch validation metrics.
        output_dir : str | Path
            Directory where AUC history plots and JSON payloads are written.
        split_name : str
            Name of the data split or fold group being summarized.
        model_scope : str
            Label describing the model or evaluation scope.
        extra_metadata : Dict[str, Any] | None
            Optional metadata added to the serialized history payload.

    Returns
    --------
        payload : Dict[str, Any]
            Serialized history overlay payload containing plotted fold summaries and skipped fold details.

    Raises
    -------
        OSError
            Raised when the output directory or artifact files cannot be created or written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "status": "ok",
        "split": split_name,
        "model_scope": model_scope,
        "n_folds_requested": int(len(fold_history_payloads))
    }
    if extra_metadata:
        payload.update(extra_metadata)

    fold_histories: list[dict[str, Any]] = []
    skipped_folds: list[dict[str, Any]] = []

    for idx, fold_payload in enumerate(fold_history_payloads):
        fold_index_raw = fold_payload.get("fold_index")
        fold_index = int(fold_index_raw) if fold_index_raw is not None else idx
        epoch_rows = fold_payload.get("epochs")
        if not isinstance(epoch_rows, list):
            skipped_folds.append(
                {"fold_index": fold_index, "reason": "missing_epochs_list"}
            )
            continue

        per_epoch: dict[int, tuple[float, float]] = {}
        for row in epoch_rows:
            if not isinstance(row, dict):
                continue
            epoch_raw = row.get("epoch")
            val_metrics = row.get("val")
            if epoch_raw is None or not isinstance(val_metrics, dict):
                continue
            try:
                epoch = int(epoch_raw)
                roc_auc = float(val_metrics["roc_auc"])
                pr_auc = float(val_metrics["pr_auc"])
            except (TypeError, ValueError, KeyError):
                continue
            if not np.isfinite(roc_auc) or not np.isfinite(pr_auc):
                continue
            per_epoch[epoch] = (roc_auc, pr_auc)

        if not per_epoch:
            skipped_folds.append(
                {"fold_index": fold_index, "reason": "no_valid_auc_history_rows"}
            )
            continue

        ordered_epochs = sorted(per_epoch.keys())
        roc_auc_values = [float(per_epoch[epoch][0])
                          for epoch in ordered_epochs]
        pr_auc_values = [float(per_epoch[epoch][1])
                         for epoch in ordered_epochs]
        fold_histories.append(
            {
                "fold_index": int(fold_index),
                "epochs": ordered_epochs,
                "roc_auc": roc_auc_values,
                "pr_auc": pr_auc_values
            }
        )

    fold_histories.sort(key=lambda row: row["fold_index"])
    payload["n_folds_plotted"] = int(len(fold_histories))
    payload["skipped_folds"] = skipped_folds
    payload["folds"] = [
        {
            "fold_index": int(row["fold_index"]),
            "n_epochs": int(len(row["epochs"])),
            "first_epoch": int(row["epochs"][0]),
            "last_epoch": int(row["epochs"][-1]),
            "roc_auc_last": float(row["roc_auc"][-1]),
            "pr_auc_last": float(row["pr_auc"][-1])
        }
        for row in fold_histories
    ]

    overlay_json_path = output_path / f"{split_name}_auc_history.json"
    if not fold_histories:
        payload["status"] = "skipped_no_valid_folds"
        payload["reason"] = "no_valid_fold_auc_histories"
        overlay_json_path.write_text(_to_json(payload), encoding="utf-8")
        return payload

    _plot_fold_auc_history(
        output_path=output_path / f"{split_name}_roc_auc_history.png",
        split_name=split_name,
        metric_name="ROC AUC",
        fold_histories=fold_histories,
        metric_key="roc_auc"
    )
    _plot_fold_auc_history(
        output_path=output_path / f"{split_name}_pr_auc_history.png",
        split_name=split_name,
        metric_name="PR AUC",
        fold_histories=fold_histories,
        metric_key="pr_auc"
    )

    overlay_json_path.write_text(_to_json(payload), encoding="utf-8")
    return payload


def _to_float_array(values: Any) -> np.ndarray:
    """Converts finite numeric values into a NumPy float array."""
    if values is None:
        return np.asarray([], dtype=float)
    cleaned = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return np.asarray(cleaned, dtype=float)


def _to_json_numbers(values: np.ndarray) -> List[float | None]:
    """Converts a NumPy array into JSON-compatible numeric values."""
    out: list[float | None] = []
    for value in values:
        val = float(value)
        out.append(val if np.isfinite(val) else None)
    return out


def _to_json(payload: Dict[str, Any]) -> str:
    """Serializes a dictionary payload as formatted JSON."""
    return json.dumps(payload, indent=2)


def _plot_roc_curve(
    output_path: Path,
    split_name: str,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float
) -> None:
    """Plots and saves a ROC curve image."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1, color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({split_name})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_pr_curve(
    output_path: Path,
    split_name: str,
    precision: np.ndarray,
    recall: np.ndarray,
    pr_auc: float,
    n_positive: int,
    n_samples: int
) -> None:
    """Plots and saves a precision-recall curve image."""
    baseline = n_positive / max(n_samples, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.axhline(
        y=baseline,
        linestyle="--",
        linewidth=1,
        color="gray",
        label=f"Prevalence = {baseline:.4f}"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve ({split_name})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_fold_roc_curves(
    output_path: Path,
    split_name: str,
    fold_curves: List[dict[str, Any]]
) -> None:
    """Plots and saves fold-level ROC curve overlays."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in fold_curves:
        fold_number = int(row["fold_index"]) + 1
        auc = float(row["roc_auc"])
        ax.plot(
            row["fpr"],
            row["tpr"],
            linewidth=2,
            label=f"Fold {fold_number} | AUC: {auc:.3f}"
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        color="#C17BB3",
        label="Chance"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Fold ROC Curves ({split_name})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_fold_pr_curves(
    output_path: Path,
    split_name: str,
    fold_curves: List[dict[str, Any]],
    prevalence: float | None
) -> None:
    """Plots and saves fold-level precision-recall curve overlays."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in fold_curves:
        fold_number = int(row["fold_index"]) + 1
        auc = float(row["pr_auc"])
        ax.plot(
            row["recall"],
            row["precision"],
            linewidth=2,
            label=f"Fold {fold_number} | AUC: {auc:.3f}"
        )

    if prevalence is not None:
        ax.axhline(
            y=prevalence,
            linestyle="--",
            linewidth=1.5,
            color="#6C757D",
            label=f"Prevalence = {prevalence:.3f}"
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Fold Precision-Recall Curves ({split_name})")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_fold_auc_history(
    output_path: Path,
    split_name: str,
    metric_name: str,
    fold_histories: List[dict[str, Any]],
    metric_key: str
) -> None:
    """Plots and saves fold-level AUC history curves."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in fold_histories:
        fold_number = int(row["fold_index"]) + 1
        epochs = row["epochs"]
        metric_values = row[metric_key]
        final_value = float(metric_values[-1])
        ax.plot(
            epochs,
            metric_values,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"Fold {fold_number} | Final: {final_value:.3f}"
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} by Epoch ({split_name})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
