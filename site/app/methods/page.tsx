import SectionBlock from "@/components/section-block";

export default function MethodsPage() {
  return (
    <>
      <section className="subhero reveal">
        <p className="hero-eyebrow">Methods</p>
        <h1>Training and Evaluation Design</h1>
        <p className="hero-copy">
          LesionShiftAI uses a shared pipeline across model families so
          cross-model comparisons remain tied to the same preprocessing,
          splitting, and evaluation protocol.
        </p>
      </section>

      <SectionBlock eyebrow="Data Flow" title="Dataset Protocol">
        <div className="card-grid">
          <article className="info-card reveal">
            <h3>Training Domain</h3>
            <p>
              ISIC 2019 samples are split into train and validation partitions
              using deterministic seeding configured in YAML.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>External Domain</h3>
            <p>
              HAM10000 is held out as an external test domain and never used
              for model selection.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Preprocessing</h3>
            <p>
              Shared image transforms and DataLoader settings ensure consistent
              feature-space assumptions across all pipelines.
            </p>
          </article>
        </div>
      </SectionBlock>

      <SectionBlock eyebrow="Models" title="Compared Pipelines">
        <div className="card-grid">
          <article className="info-card reveal">
            <h3>Baseline CNN</h3>
            <p>
              ResNet50 backbone for single-model benchmarking with direct
              validation-to-external transfer measurement.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Ensemble CNN</h3>
            <p>
              Five fold-specific ResNet50 CNN members are trained and merged via mean
              malignancy probability to test robustness under shift.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Vision Transformer (ViT-B16)</h3>
            <p>
              ViT-B16 initialized from pretrained weights with warmup and
              minimum-learning-rate control for stable fine-tuning.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Vision Transformer (ViT-L16)</h3>
            <p>
              ViT-L16 initialized from pretrained weights
              to test higher-capacity transfer under the same protocol.
            </p>
          </article>
        </div>
      </SectionBlock>

      <SectionBlock eyebrow="Evaluation" title="Metric and Artifact Policy">
        <ul className="plain-list reveal">
          <li>
            Core metrics: accuracy, precision, recall, F1, ROC AUC, PR AUC.
          </li>
          <li>
            Confusion terms: TN, FP, FN, TP are exported for each split.
          </li>
          <li>
            Curve artifacts: split-level ROC/PR PNG files and JSON payloads.
          </li>
          <li>
            Generalization gap: <code>validation - external test</code> per
            metric.
          </li>
        </ul>
      </SectionBlock>
    </>
  );
}
