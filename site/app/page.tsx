import Link from "next/link";
import SectionBlock from "@/components/section-block";
import { MODEL_RESULTS, toPercent } from "@/data/results";

const bestValidationRocModel = MODEL_RESULTS.reduce((best, candidate) =>
  candidate.validation.rocAuc > best.validation.rocAuc ? candidate : best
);

const bestExternalRocModel = MODEL_RESULTS.reduce((best, candidate) =>
  candidate.externalTest.rocAuc > best.externalTest.rocAuc ? candidate : best
);

const quickStats = [
  {
    label: "Best ISIC ROC AUC",
    value: bestValidationRocModel.validation.rocAuc.toFixed(4),
    detail: `${bestValidationRocModel.label} on internal validation`
  },
  {
    label: "Best HAM10000 ROC AUC",
    value: bestExternalRocModel.externalTest.rocAuc.toFixed(4),
    detail: `${bestExternalRocModel.label} on external test`
  },
  {
    label: "Largest Recall Drop",
    value: toPercent(
      Math.max(...MODEL_RESULTS.map((m) => m.gapValMinusTest.recall))
    ),
    detail: "Validation-to-external shift across models"
  }
];

export default function HomePage() {
  return (
    <>
      <section className="hero reveal">
        <p className="hero-eyebrow">Project Site</p>
        <h1>LesionShiftAI</h1>
        <p className="hero-copy">
          LesionShiftAI benchmarks how skin-lesion classifiers trained on ISIC
          2019 generalize to HAM10000. This site extends the repository README
          with methods, pinned results, reproducibility flow, and implementation
          context.
        </p>
        <div className="hero-links">
          <Link href="/results" className="button button-primary">
            View Results
          </Link>
          <Link href="/methods" className="button button-outline">
            Inspect Methods
          </Link>
        </div>
      </section>

      <SectionBlock eyebrow="Overview" title="Benchmark Framing">
        <div className="card-grid">
          <article className="info-card reveal">
            <h3>Problem</h3>
            <p>
              Binary skin-lesion classifiers can appear strong on internal
              validation while degrading on external datasets with different
              acquisition and patient characteristics.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Datasets</h3>
            <p>
              Training and validation are built from ISIC 2019. External
              generalization is measured on HAM10000 without domain adaptation.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Pipelines</h3>
            <p>
              Four pipelines are benchmarked: baseline CNN, 5-fold ensemble
              CNN, pretrained ViT-B16, and pretrained ViT-L16.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Artifacts</h3>
            <p>
              Each run exports checkpoints, prediction CSVs, metrics JSON, ROC
              and PR curves, and generalization-gap summaries.
            </p>
          </article>
        </div>
      </SectionBlock>

      <SectionBlock eyebrow="Pinned Snapshot" title="Key Takeaways">
        <div className="card-grid">
          {quickStats.map((stat) => (
            <article className="stat-card reveal" key={stat.label}>
              <p className="stat-label">{stat.label}</p>
              <p className="stat-value">{stat.value}</p>
              <p className="stat-detail">{stat.detail}</p>
            </article>
          ))}
        </div>
        <p className="plain-copy">
          Pinned runs: Baseline <code>20260428_042520</code>, Ensemble{" "}
          <code>ensemble_ens_20260429_b</code>, ViT-B16{" "}
          <code>20260423_105922</code>, ViT-L16{" "}
          <code>large_vit_b16_isic_to_ham</code>.
        </p>
      </SectionBlock>
    </>
  );
}
