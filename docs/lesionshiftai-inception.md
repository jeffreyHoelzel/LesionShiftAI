# LesionShiftAI Development Plan

## Objective
Build a reproducible machine learning pipeline for cross-dataset skin lesion classification comparing:
- Baseline CNN
- CNN Ensemble
- Vision Transformer (ViT)

Primary goal: evaluate generalization from ISIC → HAM10000.

---

## Project Principles
- Reproducibility first
- Consistent preprocessing
- Focus on dataset shift
- Modular design
- One-command execution

---

## Milestones

### 1. Core Infrastructure
- Config loader
- Training entry script
- Logging + checkpointing
- Deterministic seeding

Output:
outputs/{experiment_name}/

---

### 2. Data Pipeline
- ISIC + HAM10000 loaders
- Binary label mapping
- Standard preprocessing
- Dataset splits

---

### 3. Baseline CNN
- Simple CNN or pretrained backbone
- Training loop
- Metrics: accuracy, F1, ROC AUC, PR AUC

---

### 4. Evaluation Framework
- Cross-dataset evaluation
- Generalization gap tracking
- Save predictions

---

### 5. Ensemble
- Multiple CNNs
- Aggregate predictions
- Compare robustness

---

### 6. Vision Transformer
- Pretrained ViT
- Fine-tune + evaluate

---

### 7. Experiment Tracking
- Save configs + metrics
- Maintain results table

---

### 8. Error Analysis
- Analyze misclassifications
- Compare model failures

---

### 9. Testing
- Dataset tests
- Metric tests
- Reproducibility checks

---

### 10. Final Outputs
- Plots and tables
- README and docs

---

## Execution Order
1. Infrastructure
2. Data
3. CNN
4. Eval
5. Benchmark
6. Ensemble
7. ViT
8. Analysis
9. Testing
10. Polish

---

## Success Criteria
- CNN generalization performance
- Ensemble robustness gains
- ViT comparison
- Clear failure modes

---

## Notes for Codex
- Build working pipeline first
- Avoid premature complexity
- Keep modules consistent
- Log everything
