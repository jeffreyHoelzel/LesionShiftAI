import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
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
    """Write ROC/PR curve artifacts and return serialized curve payload."""
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
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write fold-level ROC/PR overlay plots from per-fold curve payloads."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "status": "ok",
        "split": split_name,
        "model_scope": model_scope,
        "n_folds_requested": int(len(fold_curve_payloads)),
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
                "precision": precision,
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
            "n_positive": int(row["n_positive"]),
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


def _to_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    cleaned = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    return np.asarray(cleaned, dtype=float)


def _to_json_numbers(values: np.ndarray) -> List[float | None]:
    out: list[float | None] = []
    for value in values:
        val = float(value)
        out.append(val if np.isfinite(val) else None)
    return out


def _to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _plot_roc_curve(
    output_path: Path,
    split_name: str,
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
) -> None:
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
    n_samples: int,
) -> None:
    baseline = n_positive / max(n_samples, 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2, label=f"PR AUC = {pr_auc:.4f}")
    ax.axhline(
        y=baseline,
        linestyle="--",
        linewidth=1,
        color="gray",
        label=f"Prevalence = {baseline:.4f}",
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
    fold_curves: List[dict[str, Any]],
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in fold_curves:
        fold_number = int(row["fold_index"]) + 1
        auc = float(row["roc_auc"])
        ax.plot(
            row["fpr"],
            row["tpr"],
            linewidth=2,
            label=f"Fold {fold_number} | AUC: {auc:.3f}",
        )

    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        color="#C17BB3",
        label="Chance",
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
    prevalence: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in fold_curves:
        fold_number = int(row["fold_index"]) + 1
        auc = float(row["pr_auc"])
        ax.plot(
            row["recall"],
            row["precision"],
            linewidth=2,
            label=f"Fold {fold_number} | AUC: {auc:.3f}",
        )

    if prevalence is not None:
        ax.axhline(
            y=prevalence,
            linestyle="--",
            linewidth=1.5,
            color="#6C757D",
            label=f"Prevalence = {prevalence:.3f}",
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
