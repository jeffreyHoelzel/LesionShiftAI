import SectionBlock from "@/components/section-block";

const snippetCnnDefinition = `class BaselineCNN(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        # single logit for BCE
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor):
        return self.backbone(x).squeeze(1)`;

const snippetVitDefinition = `class ViTBinaryClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1
        )

    def forward(self, x: torch.Tensor):
        return self.backbone(x).squeeze(-1)`;

const snippetModelSelection = `# baseline and ensemble members
model = BaselineCNN(pretrained=True).to(device)

# ViT experiment variant
model = ViTBinaryClassifier(
    model_name="vit_large_patch16_224.augreg_in21k_ft_in1k",
    pretrained=True
).to(device)`;

const snippetTrainLoop = `optimizer = AdamW(
    model.parameters(),
    lr=cfg.train.lr,
    weight_decay=cfg.train.weight_decay
)
scheduler = _build_scheduler(optimizer, cfg)

for epoch in range(start_epoch, cfg.train.epochs + 1):
    train_metrics = train_one_epoch(...)
    val_metrics, val_preds = evaluate_loader(...)
    scheduler.step()

    ckpt_payload = _build_checkpoint_payload(...)
    torch.save(ckpt_payload, run_dir / "checkpoints" / "last.pt")

    if best_pr_auc == float("-inf") or val_metrics["pr_auc"] > best_pr_auc:
        best_pr_auc = float(val_metrics["pr_auc"])
        torch.save(ckpt_payload, run_dir / "checkpoints" / "best.pt")`;

const snippetEnsembleAggregation = `test_aggregate_df = (
    all_test_preds_df
    .groupby(["dataset", "sample_id"], as_index=False)
    .agg(
        label=("label", "first"),
        prob_malignant=("prob_malignant", "mean"),
        prob_malignant_std=("prob_malignant", "std"),
        member_predictions=("member_fold", "nunique")
    )
)`;

export default function CodePage() {
  return (
    <>
      <section className="subhero reveal">
        <p className="hero-eyebrow">Code</p>
        <h1>Model Definitions and Core Snippets</h1>
        <p className="hero-copy">
          This page highlights the concrete model classes and the key places
          where they are instantiated and aggregated in benchmark runs.
        </p>
      </section>

      <SectionBlock eyebrow="Models" title="Baseline / Ensemble CNN Definition">
        <p className="plain-copy reveal">
          Source: <code>src/lesionshiftai/models/cnn.py</code>. Baseline and
          each ensemble member use the same ResNet50 backbone with a 1-logit
          classification head.
        </p>
        <pre className="code-block reveal">
          <code>{snippetCnnDefinition}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Models" title="Vision Transformer Definition">
        <p className="plain-copy reveal">
          Source: <code>src/lesionshiftai/models/vit.py</code>. The ViT wrapper
          is architecture-agnostic via <code>model_name</code> and always
          outputs a single binary logit.
        </p>
        <pre className="code-block reveal">
          <code>{snippetVitDefinition}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Training" title="Where Model Variants Are Selected">
        <p className="plain-copy reveal">
          Source: <code>scripts/train_baseline_cnn.py</code>,{" "}
          <code>scripts/train_ensemble_member_cnn.py</code>, and{" "}
          <code>scripts/train_vit.py</code>. This is where run scripts pick CNN
          vs ViT backbones and specific pretrained ViT variants.
        </p>
        <pre className="code-block reveal">
          <code>{snippetModelSelection}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Training" title="Core Fine-Tuning and Checkpoint Loop">
        <p className="plain-copy reveal">
          Source: <code>scripts/train_vit.py</code>. Each epoch performs
          train/validation passes, applies warmup+cosine scheduling, and tracks
          the best checkpoint by validation PR AUC.
        </p>
        <pre className="code-block reveal">
          <code>{snippetTrainLoop}</code>
        </pre>
      </SectionBlock>

      <SectionBlock eyebrow="Ensemble" title="How Ensemble Predictions Are Combined">
        <p className="plain-copy reveal">
          Source: <code>scripts/train_ensemble_member_cnn.py</code>. HAM10000
          aggregate predictions are formed by mean malignancy probability across
          fold members.
        </p>
        <pre className="code-block reveal">
          <code>{snippetEnsembleAggregation}</code>
        </pre>
      </SectionBlock>
    </>
  );
}
