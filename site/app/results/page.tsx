import FigureGrid from "@/components/figure-grid";
import GapTable from "@/components/gap-table";
import MetricsTable from "@/components/metrics-table";
import SectionBlock from "@/components/section-block";
import { MODEL_RESULTS } from "@/data/results";

export default function ResultsPage() {
  return (
    <>
      <section className="subhero reveal">
        <p className="hero-eyebrow">Results</p>
        <h1>Pinned Benchmark Snapshot</h1>
        <p className="hero-copy">
          This page hard-codes the v1 benchmark snapshot from pinned experiment
          runs. This page gets updated when new runs outperform old runs.
        </p>
      </section>

      <SectionBlock eyebrow="Pinned Runs" title="Run IDs Used in This Snapshot">
        <div className="card-grid">
          {MODEL_RESULTS.map((model) => (
            <article key={model.id} className="info-card reveal">
              <h3>{model.label}</h3>
              <p>
                <strong>Run ID:</strong> <code className="run-id">{model.runId}</code>
              </p>
              <p>{model.notes}</p>
            </article>
          ))}
        </div>
      </SectionBlock>

      <SectionBlock
        eyebrow="Metric Tables"
        title="Validation and External-Test Performance"
      >
        <MetricsTable
          title="ISIC Validation Metrics"
          models={MODEL_RESULTS}
          splitKey="validation"
        />
        <MetricsTable
          title="HAM10000 External-Test Metrics"
          models={MODEL_RESULTS}
          splitKey="externalTest"
        />
      </SectionBlock>

      <SectionBlock eyebrow="Shift Analysis" title="Generalization Gap Summary">
        <GapTable models={MODEL_RESULTS} />
        <p className="plain-copy reveal">
          Positive values indicate the model performed better on validation than
          on external test. This is expected under dataset shift and is one of
          the core benchmark signals.
        </p>
      </SectionBlock>

      <SectionBlock eyebrow="Curves" title="ROC and Precision-Recall Panels">
        {MODEL_RESULTS.map((model) => (
          <FigureGrid
            key={model.id}
            title={`${model.label} Curves`}
            figures={model.figures}
          />
        ))}
      </SectionBlock>
    </>
  );
}
