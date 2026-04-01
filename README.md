# LesionShiftAI

LesionShiftAI is a deep learning benchmark for cross-dataset skin lesion classification. The project evaluates how well different model families generalize when trained on one dermoscopic dataset and tested on another under real-world dataset shift.

The core experimental setting is:

- Train on ISIC 2019
- Test on HAM10000
- Compare CNN, CNN ensemble, and Vision Transformer approaches

## Project Goal

Most skin lesion classifiers are trained and evaluated on the same dataset, which can overestimate real-world performance. LesionShiftAI focuses on external generalization by measuring how well models transfer across datasets collected under different institutions, imaging conditions, and label distributions.

## Planned Model Benchmarks

- Baseline CNN
- CNN Ensemble
- Vision Transformer

## Core Questions

- How much does performance drop under dataset shift?
- Do ensembles improve robustness over a single CNN?
- Can a ViT remain competitive under limited medical imaging data?
- Which metrics matter most for clinically meaningful screening?

## Repository Structure

```text
LesionShiftAI/
├── config/        # experiment and model config files
├── docs/          # project website, reports, and supporting docs
├── notebooks/     # exploratory analysis and debugging notebooks
├── scripts/       # scripts for training, evaluation, and utilities
├── src/           # source code
│   └── lesionshiftai/
│       ├── data/      # dataset loading, preprocessing, and label mapping
│       ├── models/    # CNN, ensemble, and ViT model definitions
│       ├── train/     # training loops and optimization logic
│       ├── eval/      # metrics and cross dataset evaluation
│       └── visuals/   # plots, confusion matrices, ROC and PR curves
├── tests/         # unit and integration tests
├── pyproject.toml
└── environment.yml
```

## Datasets
### Training

#### ISIC 2019
Used as the primary training dataset. Multi class lesion labels are mapped to a binary target of benign vs malignant.

### External Testing

#### HAM10000
Used as the external test dataset to evaluate generalization under dataset shift.

## Evaluation Focus

This project emphasizes metrics that are more informative than raw accuracy for imbalanced medical classification tasks.

Primary metrics include:

- ROC AUC
- PR AUC
- Recall
- Precision
- F1 score
- Confusion matrix

Recall is especially important because missing a malignant lesion is much more costly than producing a false positive.

## Installation

### Conda environment

```bash
conda env create -f environment.yml
conda activate lesionshiftai
```

### Editable install
```bash
pip install -e .
```

## Reproducibility

This repository is being structured to support reproducible experiments through:

- version controlled configurations
- consistent label mapping across datasets
- patient-aware splitting where metadata allows
- tracked experiment outputs and evaluation artifacts

## Ethical Considerations

This project is for research and benchmarking purposes only. It is not a clinical diagnostic tool.

Important concerns include:

- false negatives in malignant lesion detection
- dataset bias across patient populations and skin tones
- reduced reliability under real world domain shift

## Current Status

Early repository setup and infrastructure development.

Planned next steps:

- data ingestion and binary label mapping
- baseline CNN training pipeline
- ensemble benchmarking
- ViT benchmarking
- cross dataset evaluation and analysis

## Author

Jeffrey Hoelzel Jr.
