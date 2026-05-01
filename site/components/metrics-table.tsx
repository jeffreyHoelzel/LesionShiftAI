import { ModelResult, SplitMetrics, toPercent } from "@/data/results";

interface MetricsTableProps {
  title: string;
  models: ModelResult[];
  splitKey: "validation" | "externalTest";
};

function metricCell(metric: number): string {
  return toPercent(metric);
}

function confusion(metrics: SplitMetrics): string {
  return `TN ${metrics.tn} | FP ${metrics.fp} | FN ${metrics.fn} | TP ${metrics.tp}`;
}

export default function MetricsTable({
  title,
  models,
  splitKey
}: MetricsTableProps) {
  return (
    <div className="table-wrap reveal">
      <h3>{title}</h3>
      <div className="table-scroll">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>ROC AUC</th>
              <th>PR AUC</th>
              <th>Confusion Matrix</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => {
              const metrics = model[splitKey];
              return (
                <tr key={`${model.id}-${splitKey}`}>
                  <th>{model.label}</th>
                  <td>{metricCell(metrics.accuracy)}</td>
                  <td>{metricCell(metrics.precision)}</td>
                  <td>{metricCell(metrics.recall)}</td>
                  <td>{metricCell(metrics.f1)}</td>
                  <td>{metrics.rocAuc.toFixed(4)}</td>
                  <td>{metrics.prAuc.toFixed(4)}</td>
                  <td>{confusion(metrics)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
