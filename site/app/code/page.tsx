import SectionBlock from "@/components/section-block";

const snippetTrainLoop = `for epoch in range(start_epoch, cfg.train.epochs + 1):
    train_metrics = train_one_epoch(...)
    val_metrics, val_preds = evaluate_loader(...)

    if best_pr_auc == float("-inf") or val_metrics["pr_auc"] > best_pr_auc:
        best_pr_auc = float(val_metrics["pr_auc"])
        torch.save(ckpt_payload, run_dir / "checkpoints" / "best.pt")`;

const snippetEval = `metrics = compute_binary_metrics(y_true_final, y_prob_final, threshold=threshold)
metrics["loss"] = loss_sum_all / max(n_all, 1)

return metrics, preds`;

const snippetGap = `keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
return {
    f"{k}_gap_val_minus_test": float(val_metrics[k] - test_metrics[k])
    for k in keys
    if k in val_metrics and k in test_metrics
}`;

export default function CodePage() {
  return (
    <>
      <section className="subhero reveal">
        <p className="hero-eyebrow">Code</p>
        <h1>Focused Implementation Snippets</h1>
        <p className="hero-copy">
          The benchmark codebase emphasizes deterministic runs, explicit metrics
          export, and external-shift analysis in each pipeline.
        </p>
      </section>

      <SectionBlock eyebrow="Training" title="Epoch Loop and Model Selection">
        <p className="plain-copy reveal">
          Each training script evaluates every epoch and tracks the best model
          by validation PR AUC before writing final split-level artifacts.
        </p>
        <pre className="code-block reveal">
          <code>{snippetTrainLoop}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Evaluation" title="Unified Metrics and Predictions">
        <p className="plain-copy reveal">
          The evaluator gathers predictions, deduplicates padded distributed
          samples, and returns both metrics and prediction rows.
        </p>
        <pre className="code-block reveal">
          <code>{snippetEval}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Shift Signal" title="Generalization Gap Computation">
        <p className="plain-copy reveal">
          Validation-minus-external deltas are computed for all core metrics and
          saved as dedicated JSON artifacts.
        </p>
        <pre className="code-block reveal">
          <code>{snippetGap}</code>
        </pre>
      </SectionBlock>
    </>
  );
}
