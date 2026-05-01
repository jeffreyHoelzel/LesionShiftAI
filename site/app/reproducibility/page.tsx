import SectionBlock from "@/components/section-block";

export default function ReproducibilityPage() {
  return (
    <>
      <section className="subhero reveal">
        <p className="hero-eyebrow">Reproducibility</p>
        <h1>Environment and Run Workflow</h1>
        <p className="hero-copy">
          The benchmark is designed for deterministic reruns with explicit
          config files, exported artifacts, and HPC-friendly launch scripts.
        </p>
      </section>

      <SectionBlock eyebrow="Local" title="Developer Setup (uv + Python ≥3.12)">
        <pre className="code-block reveal">
          <code>{`# setup local environment with UV
uv sync --extra dev --python 3.12

# run unit, integration, and coverage tests
uv run pytest -m unit --no-cov
uv run pytest -m integration --no-cov
uv run pytest

# run ViT training as an example, not recommended locally
uv run python scripts/train_vit.py --config config/vit_b16.yml --threshold 0.5`}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="HPC" title="Monsoon Runtime Flow">
        <pre className="code-block reveal">
          <code>{`# on Monsoon (HPC)
module purge
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f environment.yml -p /scratch/$USER/conda/envs/lesionshiftai
conda activate /scratch/$USER/conda/envs/lesionshiftai

# on local machine, build and transfer baseline CNN requirements as an example
uv run python scripts/build_pyz.py
scp dist/lesionshiftai.pyz <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp scripts/hpc/train_baseline_cnn.sh <USER>@monsoon.hpc.nau.edu:~/lesionshiftai
scp config/baseline_cnn.yml <USER>@monsoon.hpc.nau.edu:~/lesionshiftai`}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Launch" title="Training Jobs and Artifacts">
        <div className="card-grid">
          <article className="info-card reveal">
            <h3>Baseline</h3>
            <p>
              Submit <code>sbatch train_baseline_cnn.sh</code> to produce split
              metrics, predictions, and generalization-gap JSON.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>Ensemble</h3>
            <p>
              Set <code>ENSEMBLE_RUN_ID</code>, submit{" "}
              <code>sbatch train_ensemble_cnn.sh</code>, then inspect member and
              aggregate artifacts.
            </p>
          </article>
          <article className="info-card reveal">
            <h3>ViT</h3>
            <p>
              Submit <code>sbatch train_vit.sh</code> for ViT checkpoints,
              metrics, curves, and resume metadata.
            </p>
          </article>
        </div>
      </SectionBlock>
    </>
  );
}
