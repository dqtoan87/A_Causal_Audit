r"""
External evaluation v5 — v2 checkpoints on Fitzpatrick17k + PAD-UFES-20
=======================================================================
Replaces eval_external_v3.py for the new narrowed scope:
  - DROP  ISIC 2019 \ HAM10000 (concerns about acquisition overlap with HAM).
  - KEEP  Fitzpatrick17k (clinical photos, FST-stratified fairness audit).
  - ADD   PAD-UFES-20 (Brazilian clinical photos with metadata) — one of
          the cohorts explicitly listed by the Editor as "genuinely
          independent."

All inferences use the v2 checkpoints (Causal-Factorized = v2 baseline,
the headline model going into paper v10), NOT the v4 lam_100 winner.

Cohorts:
  A: Fitzpatrick17k subset  (~1.3k mel/bcc/akiec images, FST 1-6)
  B: PAD-UFES-20            (~2.3k clinical photos, 6 → 5 mapped HAM classes)

Subgroup analyses:
  - by Fitzpatrick skin type (both cohorts where available)
  - by anatomical site (PAD-UFES-20: HAM-mapped)
  - by sex / age (PAD-UFES-20)

Output: history_ISIC2018_v5/{fitzpatrick17k, pad_ufes_20, combined_*}
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except (RuntimeError, AttributeError):
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ham_isic_pipeline import (
    CausalFactorizedNet, ImageOnlyNet, MetaOnlyNet,
    CachedMetaDataset,
    center_crop_resize, evaluate_on_loader, compute_metrics,
    plot_confusion_matrix, plot_roc,
)


# ============================================================
# 1. Paths & global config
# ============================================================
HAM_META   = "data/HAM10000_metadata.csv"
FITZ_CSV   = "data/fitzpatrick17k/fitzpatrick17k.csv"
FITZ_IMAGES = "data/fitzpatrick17k/background removed"

PAD_DIR    = "data/PAD-UFES-20"
PAD_META   = os.path.join(PAD_DIR, "metadata.csv")
PAD_IMG_DIRS = [
    os.path.join(PAD_DIR, "imgs_part_1", "imgs_part_1"),
    os.path.join(PAD_DIR, "imgs_part_2", "imgs_part_2"),
    os.path.join(PAD_DIR, "imgs_part_3", "imgs_part_3"),
]

CKPT_DIR  = "history_ISIC2018_v2"
SAVE_ROOT = "history_ISIC2018_v5"

BATCH_SIZE  = 64
TARGET_SIZE = (224, 224)
NUM_WORKERS = int(os.environ.get("EV5_NUM_WORKERS", "0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. Reconstruct v2 categorical mappings from HAM metadata
# ============================================================
ham_df  = pd.read_csv(HAM_META)
ham_df["age"] = ham_df["age"].fillna(ham_df["age"].mean())

DX_MAP   = {dx: i for i, dx in enumerate(sorted(ham_df["dx"].unique()))}
SEX_MAP  = {"male": 0, "female": 1, "unknown": 2}
LOC_MAP  = {loc: i for i, loc in enumerate(sorted(ham_df["localization"].unique()))}
CLASS_NAMES = sorted(DX_MAP, key=DX_MAP.get)
NUM_CLASSES = len(DX_MAP)
NUM_SEX     = len(SEX_MAP)
NUM_LOC     = len(LOC_MAP)
META_DIM    = 1 + NUM_SEX + NUM_LOC
HAM_AGE_MEAN = float(ham_df["age"].mean())

print(f"v2 DX_MAP:  {DX_MAP}")
print(f"v2 LOC_MAP: {LOC_MAP}")

# ============================================================
# 3. Cohort-specific label mappings
# ============================================================
# PAD-UFES-20 diagnostic → HAM 7-class
PAD_DX_MAP = {
    "BCC": "bcc",
    "ACK": "akiec",   # actinic keratosis
    "SCC": "akiec",   # squamous cell carcinoma — closest HAM analog
    "NEV": "nv",
    "SEK": "bkl",     # seborrheic keratosis — part of HAM's bkl
    "MEL": "mel",
}

# PAD-UFES-20 region → HAM localization
PAD_LOC_MAP = {
    "FACE":     "face",
    "NOSE":     "face",
    "LIP":      "face",
    "EAR":      "ear",
    "NECK":     "neck",
    "SCALP":    "scalp",
    "CHEST":    "chest",
    "ABDOMEN":  "abdomen",
    "BACK":     "back",
    "ARM":      "upper extremity",
    "FOREARM":  "upper extremity",
    "HAND":     "hand",
    "THIGH":    "lower extremity",
    "FOOT":     "foot",
}

# Fitzpatrick17k disease → HAM (subset only)
FITZ_DX_MAP = {
    "melanoma":                "mel",
    "basal cell carcinoma":    "bcc",
    "squamous cell carcinoma": "akiec",
}


# ============================================================
# 4. Streaming dataset
# ============================================================
class StreamingImageDataset(Dataset):
    """Lazy-load images from disk; matches (img,label,age,sex,loc,age_grp)."""

    def __init__(self, df, image_paths, target_size=(224, 224)):
        self.paths       = list(image_paths)
        self.target_size = target_size
        self.to_tensor   = T.ToTensor()
        self.normalize   = T.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])

        self.labels = torch.tensor([DX_MAP[d]  for d in df["dx"]],
                                   dtype=torch.long)
        self.ages   = torch.tensor((df["age"].values / 85.0).astype("float32"))
        self.sexes  = torch.tensor([SEX_MAP[s] for s in df["sex"]],
                                   dtype=torch.long)
        self.locs   = torch.tensor([LOC_MAP[l] for l in df["localization"]],
                                   dtype=torch.long)

        ages_real = self.ages * 85.0
        self.age_groups = torch.zeros(len(self.labels), dtype=torch.long)
        self.age_groups[ages_real >= 40] = 1
        self.age_groups[ages_real >= 60] = 2

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            img = center_crop_resize(img, self.target_size)
        except Exception:
            img = Image.new("RGB", self.target_size)
        img = self.normalize(self.to_tensor(img))
        return (img, self.labels[i], self.ages[i], self.sexes[i],
                self.locs[i], self.age_groups[i])


# ============================================================
# 5. Cohort preparation
# ============================================================
def prepare_fitzpatrick17k():
    df = pd.read_csv(FITZ_CSV)
    pre = len(df)
    df = df[df["label"].isin(FITZ_DX_MAP)].copy()
    print(f"  [Fitz] kept {len(df)} of {pre} rows in {set(FITZ_DX_MAP)}")
    df["dx"] = df["label"].map(FITZ_DX_MAP)

    df["fst"] = pd.to_numeric(df["fitzpatrick_scale"],
                              errors="coerce").fillna(-1).astype(int)
    pre = len(df)
    df = df[df["fst"].between(1, 6)].copy()
    print(f"  [Fitz] kept {len(df)} of {pre} with FST ∈ [1, 6]")
    df["fst_group"] = pd.cut(df["fst"], bins=[0, 2, 4, 6],
                             labels=["FST 1-2", "FST 3-4", "FST 5-6"])

    df["image_path"] = df["md5hash"].apply(
        lambda h: os.path.join(FITZ_IMAGES, f"{h}.jpg"))
    pre = len(df)
    df = df[df["image_path"].apply(os.path.exists)].copy()
    print(f"  [Fitz] {len(df)} of {pre} images present on disk")

    df["age"] = HAM_AGE_MEAN
    df["sex"] = "unknown"
    df["localization"] = "unknown"
    return df.reset_index(drop=True)


def _find_pad_image(img_id):
    for d in PAD_IMG_DIRS:
        p = os.path.join(d, img_id)
        if os.path.exists(p):
            return p
    return None


def prepare_pad_ufes_20():
    df = pd.read_csv(PAD_META)
    print(f"  [PAD] raw rows: {len(df)}")

    # Diagnostic mapping
    pre = len(df)
    df = df[df["diagnostic"].isin(PAD_DX_MAP)].copy()
    df["dx"] = df["diagnostic"].map(PAD_DX_MAP)
    print(f"  [PAD] kept {len(df)} of {pre} mappable diagnostics")

    # Age (PAD-UFES-20 has ages 6-94; HAM is 0-85). Fill NaN with HAM mean.
    df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(HAM_AGE_MEAN)

    # Sex: gender column → HAM SEX_MAP
    df["sex"] = df["gender"].astype(str).str.lower().replace({
        "male": "male", "female": "female", "nan": "unknown"
    })
    df["sex"] = df["sex"].where(df["sex"].isin(SEX_MAP), "unknown")

    # Region → HAM localization
    df["raw_region"] = df["region"].fillna("unknown").astype(str)
    df["localization"] = df["raw_region"].map(PAD_LOC_MAP).fillna("unknown")
    df["localization"] = df["localization"].where(
        df["localization"].isin(LOC_MAP), "unknown")

    # Fitzpatrick (column name has a typo in the dataset: "fitspatrick")
    df["fst"] = pd.to_numeric(df["fitspatrick"], errors="coerce")
    df["fst_group"] = pd.cut(df["fst"], bins=[0, 2, 4, 6],
                             labels=["FST 1-2", "FST 3-4", "FST 5-6"])

    df["age_bin"] = pd.cut(df["age"], bins=[0, 40, 60, 200],
                           labels=["<40", "40-60", ">=60"])

    # Image paths
    df["image_path"] = df["img_id"].apply(_find_pad_image)
    pre = len(df)
    df = df[df["image_path"].notna()].copy()
    print(f"  [PAD] kept {len(df)} of {pre} rows with images on disk")
    return df.reset_index(drop=True)


# ============================================================
# 6. Evaluation runner
# ============================================================
def load_v2_models():
    m_img = ImageOnlyNet(NUM_CLASSES).to(DEVICE)
    m_img.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "best_image_only.pth"), map_location=DEVICE))
    m_meta = MetaOnlyNet(META_DIM, NUM_CLASSES).to(DEVICE)
    m_meta.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "best_meta_only.pth"), map_location=DEVICE))
    m_caus = CausalFactorizedNet(NUM_CLASSES, NUM_SEX, NUM_LOC).to(DEVICE)
    m_caus.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "best_causal_factorized.pth"),
        map_location=DEVICE))
    m_img.eval(); m_meta.eval(); m_caus.eval()
    return m_img, m_meta, m_caus


def eval_models(df, save_dir, prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    img_ds = StreamingImageDataset(df, df["image_path"].values,
                                   target_size=TARGET_SIZE)
    img_ld = DataLoader(img_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    meta_dict = {"labels": img_ds.labels, "ages": img_ds.ages,
                 "sexes": img_ds.sexes, "locs": img_ds.locs}
    meta_ds = CachedMetaDataset(meta_dict, NUM_SEX, NUM_LOC)
    meta_ld = DataLoader(meta_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

    m_img, m_meta, m_caus = load_v2_models()

    results = {}
    for name, mdl, ld, mtype, key in [
        ("Image-Only",        m_img,  img_ld,  "image_only",       "image_only"),
        ("Meta-Only",         m_meta, meta_ld, "meta_only",        "meta_only"),
        ("Causal-Factorized", m_caus, img_ld,  "causal_factorized","causal_factorized"),
    ]:
        print(f"  [{prefix}] {name} …", flush=True)
        preds, probs, labels = evaluate_on_loader(mdl, ld, DEVICE, mtype)
        acc, cm, ss = compute_metrics(preds, labels, NUM_CLASSES)
        ss["class_name"] = [CLASS_NAMES[i] for i in ss["class_idx"]]
        results[name] = dict(acc=acc, cm=cm, ss=ss,
                             preds=preds, probs=probs, labels=labels)
        print(f"    Acc={acc:.4f}  Sens={ss['sensitivity'].mean():.4f}  "
              f"Spec={ss['specificity'].mean():.4f}")
        plot_confusion_matrix(cm, CLASS_NAMES, f"{prefix} CM — {name}",
                              os.path.join(save_dir, f"cm_{key}.png"))
        plot_roc(probs, labels, NUM_CLASSES, CLASS_NAMES,
                 f"{prefix} ROC — {name}",
                 os.path.join(save_dir, f"roc_{key}.png"))
        ss.to_csv(os.path.join(save_dir, f"sens_spec_{key}.csv"), index=False)

    summary = pd.DataFrame([
        {"Model": n, "N": int(len(r["labels"])),
         "Accuracy": r["acc"],
         "Mean Sensitivity": r["ss"]["sensitivity"].mean(),
         "Mean Specificity": r["ss"]["specificity"].mean()}
        for n, r in results.items()
    ])
    summary.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    print("\n", summary.to_string(index=False))
    return results, summary


# ============================================================
# 7. Subgroup analysis (with melanoma-specific sensitivity)
# ============================================================
def subgroup_eval(results, df, group_col, save_dir, suffix=""):
    rows = []
    mel_idx = DX_MAP.get("mel")
    bcc_idx = DX_MAP.get("bcc")
    for name, r in results.items():
        if name == "Meta-Only":
            continue
        for g in df[group_col].dropna().unique():
            mask = (df[group_col].astype(str) == str(g)).values
            n = int(mask.sum())
            if n < 5:
                continue
            sub_acc = float((r["preds"][mask] == r["labels"][mask]).mean())
            mel_mask = mask & (r["labels"] == mel_idx)
            mel_sens = (float((r["preds"][mel_mask]
                               == r["labels"][mel_mask]).mean())
                        if mel_mask.sum() > 0 else float("nan"))
            bcc_mask = mask & (r["labels"] == bcc_idx)
            bcc_sens = (float((r["preds"][bcc_mask]
                               == r["labels"][bcc_mask]).mean())
                        if bcc_mask.sum() > 0 else float("nan"))
            rows.append({
                "Model": name, "Group": str(g), "N": n,
                "Accuracy": sub_acc,
                "Mel N": int(mel_mask.sum()), "Mel Sensitivity": mel_sens,
                "Bcc N": int(bcc_mask.sum()), "Bcc Sensitivity": bcc_sens,
            })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(save_dir, f"subgroup_{group_col}{suffix}.csv"),
               index=False)
    print(f"\n  Subgroup ({group_col}):")
    print(out.to_string(index=False))
    return out


# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":
    os.makedirs(SAVE_ROOT, exist_ok=True)
    print(f"Device: {DEVICE}\n")

    # ---------- Cohort A: Fitzpatrick17k ----------
    print("=" * 60)
    print("COHORT A: Fitzpatrick17k (clinical photos, FST 1-6)")
    print("=" * 60)
    fitz_dir = os.path.join(SAVE_ROOT, "fitzpatrick17k")
    fitz_df  = prepare_fitzpatrick17k()
    print(f"\n  Final eval set: N={len(fitz_df)}")
    print(f"  dx distribution: "
          f"{dict(fitz_df['dx'].value_counts())}")
    print(f"  FST distribution: "
          f"{dict(fitz_df['fst_group'].value_counts(sort=False))}")

    fitz_results, fitz_summary = eval_models(
        fitz_df, fitz_dir, prefix="Fitz17k")
    subgroup_eval(fitz_results, fitz_df, "fst_group", fitz_dir)
    subgroup_eval(fitz_results, fitz_df, "dx",        fitz_dir)

    # ---------- Cohort B: PAD-UFES-20 ----------
    print("\n" + "=" * 60)
    print("COHORT B: PAD-UFES-20 (Brazilian clinical photos, full metadata)")
    print("=" * 60)
    pad_dir = os.path.join(SAVE_ROOT, "pad_ufes_20")
    pad_df  = prepare_pad_ufes_20()
    print(f"\n  Final eval set: N={len(pad_df)}")
    print(f"  dx distribution: "
          f"{dict(pad_df['dx'].value_counts())}")
    print(f"  sex distribution: "
          f"{dict(pad_df['sex'].value_counts())}")
    print(f"  localization distribution:")
    for k, v in pad_df["localization"].value_counts().items():
        print(f"    {k}: {v}")
    print(f"  FST distribution (non-NaN): "
          f"{dict(pad_df['fst_group'].value_counts(sort=False, dropna=True))}")

    pad_results, pad_summary = eval_models(
        pad_df, pad_dir, prefix="PAD-UFES-20")
    subgroup_eval(pad_results, pad_df, "fst_group",    pad_dir)
    subgroup_eval(pad_results, pad_df, "localization", pad_dir)
    subgroup_eval(pad_results, pad_df, "sex",          pad_dir)
    subgroup_eval(pad_results, pad_df, "age_bin",      pad_dir)
    subgroup_eval(pad_results, pad_df, "dx",           pad_dir)

    # ---------- Combined ----------
    print("\n" + "=" * 60)
    print("COMBINED: HAM val + ISIC2018 Test + Fitz + PAD-UFES-20")
    print("=" * 60)
    rows = []

    v2_val   = pd.read_csv("history_ISIC2018_v2/comparison_summary.csv")
    v2_ext   = pd.read_csv("history_ISIC2018_v2/ext_test_summary.csv")
    v2_extkn = pd.read_csv("history_ISIC2018_v2/ext_test_summary_no_unknown.csv")

    for _, r in v2_val.iterrows():
        rows.append({"Cohort": "HAM val (in-dist)", "Model": r["Model"],
                     "N": "—", "Accuracy": r["Val Accuracy"],
                     "Mean Sens": r["Mean Sensitivity"],
                     "Mean Spec": r["Mean Specificity"]})
    for _, r in v2_ext.iterrows():
        rows.append({"Cohort": "ISIC2018 Test (full)", "Model": r["Model"],
                     "N": 1511, "Accuracy": r["Ext Test Accuracy"],
                     "Mean Sens": r["Mean Sensitivity"],
                     "Mean Spec": r["Mean Specificity"]})
    for _, r in v2_extkn.iterrows():
        rows.append({"Cohort": "ISIC2018 Test (known-only)",
                     "Model": r["Model"], "N": int(r["N"]),
                     "Accuracy": r["Ext Test Accuracy (known-only)"],
                     "Mean Sens": r["Mean Sensitivity (known-only)"],
                     "Mean Spec": r["Mean Specificity (known-only)"]})
    for _, r in fitz_summary.iterrows():
        rows.append({"Cohort": "Fitzpatrick17k (mel/bcc/scc)",
                     "Model": r["Model"], "N": int(r["N"]),
                     "Accuracy": r["Accuracy"],
                     "Mean Sens": r["Mean Sensitivity"],
                     "Mean Spec": r["Mean Specificity"]})
    for _, r in pad_summary.iterrows():
        rows.append({"Cohort": "PAD-UFES-20",
                     "Model": r["Model"], "N": int(r["N"]),
                     "Accuracy": r["Accuracy"],
                     "Mean Sens": r["Mean Sensitivity"],
                     "Mean Spec": r["Mean Specificity"]})

    combo = pd.DataFrame(rows)
    combo.to_csv(os.path.join(SAVE_ROOT, "combined_summary.csv"), index=False)
    print(combo.to_string(index=False))

    # Bar plot
    cohorts = list(combo["Cohort"].unique())
    models  = ["Image-Only", "Meta-Only", "Causal-Factorized"]
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(cohorts))
    w = 0.27
    palette = {"Image-Only":        "#4C72B0",
               "Meta-Only":         "#55A868",
               "Causal-Factorized": "#C44E52"}
    for i, m in enumerate(models):
        accs = []
        for c in cohorts:
            sub = combo[(combo["Cohort"] == c) & (combo["Model"] == m)]
            accs.append(float(sub["Accuracy"].iloc[0]) if len(sub) else 0)
        bars = ax.bar(x + (i - 1) * w, accs, w, label=m, color=palette[m])
        for b, a in zip(bars, accs):
            ax.text(b.get_x() + b.get_width() / 2, a + 0.005,
                    f"{a:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, rotation=15, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Cross-cohort accuracy — v2 checkpoints")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_ROOT, "combined_accuracy.png"), dpi=150)
    plt.close()

    print(f"\nAll outputs in {SAVE_ROOT}/")
