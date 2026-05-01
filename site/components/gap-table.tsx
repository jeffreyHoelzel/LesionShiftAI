import { ModelResult } from "@/data/results";

interface GapTableProps {
  models: ModelResult[];
};

function signedPercent(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}%`;
}

export default function GapTable({ models }: GapTableProps) {
  return (
    <div className="table-wrap reveal">
      <h3>Generalization Gap (Validation - External Test)</h3>
      <div className="table-scroll">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy Gap</th>
              <th>Precision Gap</th>
              <th>Recall Gap</th>
              <th>F1 Gap</th>
              <th>ROC AUC Gap</th>
              <th>PR AUC Gap</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={`${model.id}-gap`}>
                <th>{model.label}</th>
                <td>{signedPercent(model.gapValMinusTest.accuracy)}</td>
                <td>{signedPercent(model.gapValMinusTest.precision)}</td>
                <td>{signedPercent(model.gapValMinusTest.recall)}</td>
                <td>{signedPercent(model.gapValMinusTest.f1)}</td>
                <td>{signedPercent(model.gapValMinusTest.rocAuc)}</td>
                <td>{signedPercent(model.gapValMinusTest.prAuc)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
