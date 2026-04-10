import argparse
from lesionshiftai.core.config import load_config
from lesionshiftai.core.reproducibility import set_seed
from lesionshiftai.data.datamodule import binary_counts, build_data_bundle


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/baseline_cnn.yml", type=str)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed, cfg.deterministic)

    bundle = build_data_bundle(cfg)

    print("train:", len(bundle.train_df), binary_counts(bundle.train_df))
    print("val:", len(bundle.val_df), binary_counts(bundle.val_df))
    print("test:", len(bundle.test_df), binary_counts(bundle.test_df))

    batch = next(iter(bundle.train_loader))
    print("batch image shape:", tuple(batch["image"].shape))
    print("batch label shape:", tuple(batch["label"].shape))


if __name__ == "__main__":
    main()
