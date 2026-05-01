export interface SplitMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  prAuc: number;
  tn: number;
  fp: number;
  fn: number;
  tp: number;
};

export interface GapMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  rocAuc: number;
  prAuc: number;
};

export interface FigureRef {
  title: string;
  path: string;
  caption: string;
};

export interface ModelResult {
  id: "baseline" | "ensemble" | "vit" | "vitLarge";
  label: string;
  runId: string;
  notes: string;
  validation: SplitMetrics;
  externalTest: SplitMetrics;
  gapValMinusTest: GapMetrics;
  figures: FigureRef[];
};

export const PINNED_RUNS = {
  baseline: "20260428_042520",
  ensemble: "ensemble_ens_20260429_b",
  vit: "20260423_105922",
  vitLarge: "20260430_165245"
} as const;

export const MODEL_RESULTS: ModelResult[] = [
  {
    id: "baseline",
    label: "Baseline CNN",
    runId: PINNED_RUNS.baseline,
    notes: "Single-model ResNet50 baseline with a fixed 0.5 threshold.",
    validation: {
      accuracy: 0.8513913558318532,
      precision: 0.5556368960468521,
      recall: 0.8386740331491712,
      f1: 0.6684280052840158,
      rocAuc: 0.929241147875676,
      prAuc: 0.8063536938348496,
      tn: 3555,
      fp: 607,
      fn: 146,
      tp: 759
    },
    externalTest: {
      accuracy: 0.8434348477284074,
      precision: 0.6176829268292683,
      recall: 0.5184237461617196,
      f1: 0.563717306622148,
      rocAuc: 0.796491554862444,
      prAuc: 0.6262503584238894,
      tn: 7434,
      fp: 627,
      fn: 941,
      tp: 1013
    },
    gapValMinusTest: {
      accuracy: 0.007956508103445836,
      precision: -0.06204603078241622,
      recall: 0.3202502869874516,
      f1: 0.10471069866186777,
      rocAuc: 0.13274959301323197,
      prAuc: 0.18010333541096013
    },
    figures: [
      {
        title: "ISIC Validation ROC",
        path: "/results/baseline/val_final_roc.png",
        caption: "Validation ROC curve on the ISIC split."
      },
      {
        title: "ISIC Validation PR",
        path: "/results/baseline/val_final_pr.png",
        caption: "Validation precision-recall curve on ISIC."
      },
      {
        title: "HAM10000 ROC",
        path: "/results/baseline/ham_test_roc.png",
        caption: "External-test ROC curve on HAM10000."
      },
      {
        title: "HAM10000 PR",
        path: "/results/baseline/ham_test_pr.png",
        caption: "External-test precision-recall curve on HAM10000."
      }
    ]
  },
  {
    id: "ensemble",
    label: "Ensemble CNN (5-fold)",
    runId: PINNED_RUNS.ensemble,
    notes: "Five member folds aggregated via mean malignancy probability.",
    validation: {
      accuracy: 0.8341222879684418,
      precision: 0.5278260869565218,
      recall: 0.6707182320441989,
      f1: 0.5907542579075425,
      rocAuc: 0.8527123556604961,
      prAuc: 0.6510786659776723,
      tn: 3622,
      fp: 543,
      fn: 298,
      tp: 607
    },
    externalTest: {
      accuracy: 0.8369445831253121,
      precision: 0.6274821286735505,
      recall: 0.40429887410440124,
      f1: 0.4917522564581388,
      rocAuc: 0.7966212593153255,
      prAuc: 0.5738509154865409,
      tn: 7592,
      fp: 469,
      fn: 1164,
      tp: 790
    },
    gapValMinusTest: {
      accuracy: -0.0028222951568702692,
      precision: -0.09965604171702869,
      recall: 0.26641935793979765,
      f1: 0.09900200144940374,
      rocAuc: 0.05609109634517062,
      prAuc: 0.0772277504911314
    },
    figures: [
      {
        title: "ISIC Aggregate ROC",
        path: "/results/ensemble/isic_val_member_folds_roc.png",
        caption: "ROC curves of the 5-folds on ISIC validation."
      },
      {
        title: "ISIC Aggregate PR",
        path: "/results/ensemble/isic_val_member_folds_pr.png",
        caption: "Precision-recall curves of the ISIC 5-folds."
      },
      {
        title: "HAM10000 Aggregate ROC",
        path: "/results/ensemble/ham_test_member_folds_roc.png",
        caption: "External-test ROC fold curves of the ensemble."
      },
      {
        title: "HAM10000 Aggregate PR",
        path: "/results/ensemble/ham_test_member_folds_pr.png",
        caption: "External-test precision-recall fold curves of the ensemble."
      }
    ]
  },
  {
    id: "vit",
    label: "Vision Transformer (ViT-B16)",
    runId: PINNED_RUNS.vit,
    notes: "Pretrained ViT with warmup and cosine-style learning-rate decay.",
    validation: {
      accuracy: 0.9240181567002171,
      precision: 0.7981651376146789,
      recall: 0.7690607734806629,
      f1: 0.7833427124366911,
      rocAuc: 0.9553784437464988,
      prAuc: 0.8713157165758099,
      tn: 3986,
      fp: 176,
      fn: 209,
      tp: 696
    },
    externalTest: {
      accuracy: 0.8895656515227159,
      precision: 0.9100580270793037,
      recall: 0.48157625383828045,
      f1: 0.6298527443105756,
      rocAuc: 0.8469713153174293,
      prAuc: 0.7059121438139895,
      tn: 7968,
      fp: 93,
      fn: 1013,
      tp: 941
    },
    gapValMinusTest: {
      accuracy: 0.0344525051775012,
      precision: -0.11189288946462483,
      recall: 0.2874845196423825,
      f1: 0.15348996812611548,
      rocAuc: 0.10840712842906952,
      prAuc: 0.16540357276182038
    },
    figures: [
      {
        title: "ISIC Validation ROC",
        path: "/results/vit/val_final_roc.png",
        caption: "Validation ROC curve for ViT-B16 on ISIC."
      },
      {
        title: "ISIC Validation PR",
        path: "/results/vit/val_final_pr.png",
        caption: "Validation precision-recall curve for ViT-B16."
      },
      {
        title: "HAM10000 ROC",
        path: "/results/vit/ham_test_roc.png",
        caption: "External-test ROC curve for ViT-B16 on HAM10000."
      },
      {
        title: "HAM10000 PR",
        path: "/results/vit/ham_test_pr.png",
        caption: "External-test precision-recall curve for ViT-B16."
      }
    ]
  },
  {
    id: "vitLarge",
    label: "Vision Transformer (ViT-L16)",
    runId: PINNED_RUNS.vitLarge,
    notes: "Pretrained larger ViT with warmup and cosine-style learning-rate decay.",
    validation: {
      accuracy: 0.9386224590487467,
      precision: 0.8445475638051044,
      recall: 0.8044198895027624,
      f1: 0.8239954725523486,
      rocAuc: 0.9657982907707461,
      prAuc: 0.9013507949654285,
      tn: 4028,
      fp: 134,
      fn: 177,
      tp: 728
    },
    externalTest: {
      accuracy: 0.8984523215177234,
      precision: 0.9500480307396734,
      recall: 0.5061412487205732,
      f1: 0.6604340567612688,
      rocAuc: 0.835486789128494,
      prAuc: 0.7158678083129212,
      tn: 8009,
      fp: 52,
      fn: 965,
      tp: 989
    },
    gapValMinusTest: {
      accuracy: 0.04017013753102339,
      precision: -0.10550046693456894,
      recall: 0.29827864078218924,
      f1: 0.16356141579107986,
      rocAuc: 0.13031150164225214,
      prAuc: 0.1854829866525073
    },
    figures: [
      {
        title: "ISIC Validation ROC",
        path: "/results/vit_large/val_final_roc.png",
        caption: "Validation ROC curve for ViT-L16 on ISIC."
      },
      {
        title: "ISIC Validation PR",
        path: "/results/vit_large/val_final_pr.png",
        caption: "Validation precision-recall curve for ViT-L16."
      },
      {
        title: "HAM10000 ROC",
        path: "/results/vit_large/ham_test_roc.png",
        caption: "External-test ROC curve for ViT-L16 on HAM10000."
      },
      {
        title: "HAM10000 PR",
        path: "/results/vit_large/ham_test_pr.png",
        caption: "External-test precision-recall curve for ViT-L16."
      }
    ]
  }
];

export function toPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}
