"""
v4 train — multi-seed for confidence intervals
==============================================
Re-runs the v2 protocol (3 models × 30 epochs) for 4 additional seeds.
Combined with v2 (seed=42) this gives N=5 for confidence intervals on
the headline numbers (HAM val, ISIC2018 Test, Fitzpatrick17k, PAD-UFES-20).

For each seed s in [0, 1, 7, 123]:
  history_ISIC2018_v6/seed_{s}/
     best_image_only.pth
     best_meta_only.pth
     best_causal_factorized.pth
     history_*.csv
     summary_val.csv          (HAM val, 3 models)
     summary_isic2018_test.csv
     summary_fitzpatrick17k.csv
     summary_pad_ufes_20.csv

Run aggregate_seeds.py afterwards to compute mean ± SD across seeds.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# This box has 256 logical CPUs; PyTorch defaults its intra-op pool to
# ~128 threads. The CPU-side data augmentation (RandomRotation interpolation,
# ColorJitter) is small per call, so 128-way intra-op parallelism is pure
# synchronization overhead — benchmarked at 6979 ms/batch vs 402 ms/batch at
# 8 threads (~17x). Capping the pool keeps augmentation off the critical path.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "8")))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ham_isic_pipeline import (
    CausalFactorizedNet, ImageOnlyNet, MetaOnlyNet,
    CachedImageDataset, CachedMetaDataset,
    preprocess_and_cache, preprocess_test_cache,
    train_image_only, train_meta_only, train_causal_factorized,
    evaluate_on_loader, compute_metrics,
    center_crop_resize,
)
from external_cohorts import (
    prepare_fitzpatrick17k, prepare_pad_ufes_20,
    StreamingImageDataset,
)


# ============================================================
# Config
# ============================================================
# Full 10-seed run for the n=10 multi-seed analysis. All seeds are
# trained from scratch under the SAME 30-epoch protocol (including
# seed 42, so the protocol is uniform across all 10 — previously
# seed 42 lived in history_ISIC2018_v2 at a different epoch budget).
# Each seed writes to history_ISIC2018_v6/seed_{s}/.
SEEDS_TO_RUN = [0, 1, 2, 3, 7, 11, 31, 42, 99, 123]
EPOCHS_PER_MODEL = 30
BATCH_SIZE  = 32
LR_IMAGE    = 1e-4
LR_META     = 1e-3
LR_CAUSAL   = 1e-4
LAMBDA_META = 0.1
LAMBDA_ORTH = 0.01     # v2 baseline (NOT the v4 lam_100 value)
LAMBDA_IRM  = 1.0
LAMBDA_MINV = 0.1
IRM_WARMUP  = 5
TARGET_SIZE = (224, 224)

IMAGE_DIR      = "data/HAM10000_images_combined_600x450"
SEG_DIR        = "data/HAM10000_segmentations_lesion_tschandl"
META_CSV       = "data/HAM10000_metadata.csv"
TEST_IMAGE_DIR = "data/ISIC2018_Task3_Test_Images"
TEST_GT_CSV    = "data/ISIC2018_Task3_Test_GroundTruth.csv"

SAVE_ROOT = "history_ISIC2018_v6"
CACHE_ROOT = "cache_ISIC2018_seed"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))

# ============================================================
# Mappings (must match v2 to allow checkpoint loading later)
# ============================================================
ham_df_full = pd.read_csv(META_CSV)
ham_df_full["age"] = ham_df_full["age"].fillna(ham_df_full["age"].mean())
DX_MAP   = {dx: i for i, dx in enumerate(sorted(ham_df_full["dx"].unique()))}
SEX_MAP  = {"male": 0, "female": 1, "unknown": 2}
LOC_MAP  = {loc: i for i, loc in enumerate(sorted(ham_df_full["localization"].unique()))}
CLASS_NAMES = sorted(DX_MAP, key=DX_MAP.get)
NUM_CLASSES = len(DX_MAP); NUM_SEX = len(SEX_MAP); NUM_LOC = len(LOC_MAP)
META_DIM = 1 + NUM_SEX + NUM_LOC


# ============================================================
# Helpers
# ============================================================
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_in_domain(models_dict, val_img_loader, val_meta_loader, num_classes):
    """Returns DataFrame of (Model, Accuracy, Mean Sens, Mean Spec)."""
    rows = []
    for name, (mdl, mtype, ld) in models_dict.items():
        preds, _, labels = evaluate_on_loader(mdl, ld, DEVICE, mtype)
        acc, _, ss = compute_metrics(preds, labels, num_classes)
        rows.append({
            "Model": name,
            "Accuracy": acc,
            "Mean Sensitivity": ss["sensitivity"].mean(),
            "Mean Specificity": ss["specificity"].mean(),
        })
    return pd.DataFrame(rows)


def eval_streaming_cohort(df, models_loaded, num_classes, batch_size=64):
    """Run inference on a streaming-image cohort (Fitz / PAD)."""
    img_ds = StreamingImageDataset(df, df["image_path"].values,
                                   target_size=TARGET_SIZE)
    img_ld = DataLoader(img_ds, batch_size=batch_size, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    meta_dict = {"labels": img_ds.labels, "ages": img_ds.ages,
                 "sexes": img_ds.sexes, "locs": img_ds.locs}
    meta_ds = CachedMetaDataset(meta_dict, NUM_SEX, NUM_LOC)
    meta_ld = DataLoader(meta_ds, batch_size=batch_size, shuffle=False)

    rows = []
    for name, (mdl, mtype) in models_loaded.items():
        ld = meta_ld if mtype == "meta_only" else img_ld
        preds, _, labels = evaluate_on_loader(mdl, ld, DEVICE, mtype)
        acc, _, ss = compute_metrics(preds, labels, num_classes)
        rows.append({
            "Model": name, "N": int(len(labels)),
            "Accuracy": acc,
            "Mean Sensitivity": ss["sensitivity"].mean(),
            "Mean Specificity": ss["specificity"].mean(),
        })
    return pd.DataFrame(rows)


# ============================================================
# Per-seed pipeline
# ============================================================
def run_one_seed(seed, fitz_df_template, pad_df_template, test_df_template):
    """Train 3 models for one seed and evaluate on all cohorts."""
    print("\n" + "#" * 70)
    print(f"# SEED = {seed}")
    print("#" * 70)
    t0 = time.time()

    save_dir = os.path.join(SAVE_ROOT, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    cache_dir = f"{CACHE_ROOT}_{seed}"
    os.makedirs(cache_dir, exist_ok=True)

    set_seeds(seed)

    # ---- Split with this seed ----
    train_df, val_df = train_test_split(
        ham_df_full, test_size=0.2, stratify=ham_df_full["dx"],
        random_state=seed)

    # ---- Preprocess (per-seed cache) ----
    train_cache = os.path.join(cache_dir, "train_data.pt")
    val_cache   = os.path.join(cache_dir, "val_data.pt")
    train_data = preprocess_and_cache(
        train_df, IMAGE_DIR, SEG_DIR, DX_MAP, SEX_MAP, LOC_MAP,
        TARGET_SIZE, train_cache)
    val_data   = preprocess_and_cache(
        val_df, IMAGE_DIR, SEG_DIR, DX_MAP, SEX_MAP, LOC_MAP,
        TARGET_SIZE, val_cache)

    train_aug = T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_aug = T.Compose([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_ds  = CachedImageDataset(train_data, transform=train_aug)
    val_ds    = CachedImageDataset(val_data,   transform=val_aug)
    train_img_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True)
    val_img_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)

    train_meta_ds = CachedMetaDataset(train_data, NUM_SEX, NUM_LOC)
    val_meta_ds   = CachedMetaDataset(val_data,   NUM_SEX, NUM_LOC)
    train_meta_loader = DataLoader(train_meta_ds, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0, pin_memory=True)
    val_meta_loader   = DataLoader(val_meta_ds, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=0, pin_memory=True)

    # ---- 1) Image-Only ----
    print(f"\n  [seed {seed}] training Image-Only ({EPOCHS_PER_MODEL} ep)…")
    set_seeds(seed)
    m_img = ImageOnlyNet(NUM_CLASSES).to(DEVICE)
    hist_img, best_img = train_image_only(
        m_img, train_img_loader, val_img_loader, DEVICE,
        epochs=EPOCHS_PER_MODEL, lr=LR_IMAGE,
        save_path=os.path.join(save_dir, "best_image_only.pth"))
    pd.DataFrame(hist_img).to_csv(
        os.path.join(save_dir, "history_image_only.csv"), index=False)

    # ---- 2) Meta-Only ----
    print(f"\n  [seed {seed}] training Meta-Only ({EPOCHS_PER_MODEL} ep)…")
    set_seeds(seed)
    m_meta = MetaOnlyNet(meta_dim=META_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    hist_meta, best_meta = train_meta_only(
        m_meta, train_meta_loader, val_meta_loader, DEVICE,
        epochs=EPOCHS_PER_MODEL, lr=LR_META,
        save_path=os.path.join(save_dir, "best_meta_only.pth"))
    pd.DataFrame(hist_meta).to_csv(
        os.path.join(save_dir, "history_meta_only.csv"), index=False)

    # ---- 3) Causal-Factorized ----
    print(f"\n  [seed {seed}] training Causal-Factorized ({EPOCHS_PER_MODEL} ep)…")
    set_seeds(seed)
    m_caus = CausalFactorizedNet(
        num_classes=NUM_CLASSES, num_sex=NUM_SEX, num_loc=NUM_LOC).to(DEVICE)
    hist_caus, best_caus = train_causal_factorized(
        m_caus, train_img_loader, val_img_loader, DEVICE,
        epochs=EPOCHS_PER_MODEL, lr=LR_CAUSAL,
        lambda_meta=LAMBDA_META, lambda_orth=LAMBDA_ORTH,
        lambda_irm=LAMBDA_IRM, lambda_minv=LAMBDA_MINV,
        irm_warmup_epochs=IRM_WARMUP,
        save_path=os.path.join(save_dir, "best_causal_factorized.pth"))
    pd.DataFrame(hist_caus).to_csv(
        os.path.join(save_dir, "history_causal_factorized.csv"), index=False)

    # ---- Reload best checkpoints ----
    m_img.load_state_dict(torch.load(
        os.path.join(save_dir, "best_image_only.pth"), map_location=DEVICE))
    m_meta.load_state_dict(torch.load(
        os.path.join(save_dir, "best_meta_only.pth"), map_location=DEVICE))
    m_caus.load_state_dict(torch.load(
        os.path.join(save_dir, "best_causal_factorized.pth"),
        map_location=DEVICE))
    m_img.eval(); m_meta.eval(); m_caus.eval()

    # ---- Eval on HAM val ----
    print(f"\n  [seed {seed}] evaluating on HAM val…")
    val_summary = eval_in_domain(
        {"Image-Only":        (m_img,  "image_only",        val_img_loader),
         "Meta-Only":         (m_meta, "meta_only",         val_meta_loader),
         "Causal-Factorized": (m_caus, "causal_factorized", val_img_loader)},
        val_img_loader, val_meta_loader, NUM_CLASSES)
    val_summary.to_csv(os.path.join(save_dir, "summary_val.csv"), index=False)
    print(val_summary.to_string(index=False))

    # ---- Eval on ISIC2018 Test ----
    print(f"\n  [seed {seed}] evaluating on ISIC2018 Test…")
    test_cache = os.path.join(cache_dir, "test_external_data.pt")
    test_data = preprocess_test_cache(
        test_df_template, TEST_IMAGE_DIR, DX_MAP, SEX_MAP, LOC_MAP,
        TARGET_SIZE, test_cache)
    test_ds = CachedImageDataset(test_data, transform=val_aug)
    test_img_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)
    test_meta_ds = CachedMetaDataset(test_data, NUM_SEX, NUM_LOC)
    test_meta_loader = DataLoader(test_meta_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, pin_memory=True)
    test_summary = eval_in_domain(
        {"Image-Only":        (m_img,  "image_only",        test_img_loader),
         "Meta-Only":         (m_meta, "meta_only",         test_meta_loader),
         "Causal-Factorized": (m_caus, "causal_factorized", test_img_loader)},
        test_img_loader, test_meta_loader, NUM_CLASSES)
    test_summary.to_csv(os.path.join(save_dir, "summary_isic2018_test.csv"),
                        index=False)
    print(test_summary.to_string(index=False))

    # Known-only sub-eval
    sexes_ext = test_data["sexes"].numpy()
    locs_ext  = test_data["locs"].numpy()
    unk_sex_idx = SEX_MAP["unknown"]
    unk_loc_idx = LOC_MAP.get("unknown", -1)
    known_mask = (sexes_ext != unk_sex_idx) & (locs_ext != unk_loc_idx)
    if known_mask.sum() > 20:
        rows = []
        for name, mdl, mtype, ld in [
            ("Image-Only", m_img, "image_only", test_img_loader),
            ("Meta-Only",  m_meta, "meta_only", test_meta_loader),
            ("Causal-Factorized", m_caus, "causal_factorized", test_img_loader),
        ]:
            preds, _, labels = evaluate_on_loader(mdl, ld, DEVICE, mtype)
            preds_k = preds[known_mask]
            labels_k = labels[known_mask]
            acc_k, _, ss_k = compute_metrics(preds_k, labels_k, NUM_CLASSES)
            rows.append({
                "Model": name, "N": int(known_mask.sum()),
                "Accuracy": acc_k,
                "Mean Sensitivity": ss_k["sensitivity"].mean(),
                "Mean Specificity": ss_k["specificity"].mean(),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(save_dir, "summary_isic2018_test_known_only.csv"),
            index=False)

    # ---- Models for streaming eval ----
    models_loaded = {
        "Image-Only":        (m_img,  "image_only"),
        "Meta-Only":         (m_meta, "meta_only"),
        "Causal-Factorized": (m_caus, "causal_factorized"),
    }

    # ---- Eval on Fitzpatrick17k ----
    print(f"\n  [seed {seed}] evaluating on Fitzpatrick17k…")
    fitz_summary = eval_streaming_cohort(
        fitz_df_template, models_loaded, NUM_CLASSES)
    fitz_summary.to_csv(
        os.path.join(save_dir, "summary_fitzpatrick17k.csv"), index=False)
    print(fitz_summary.to_string(index=False))

    # ---- Eval on PAD-UFES-20 ----
    print(f"\n  [seed {seed}] evaluating on PAD-UFES-20…")
    pad_summary = eval_streaming_cohort(
        pad_df_template, models_loaded, NUM_CLASSES)
    pad_summary.to_csv(
        os.path.join(save_dir, "summary_pad_ufes_20.csv"), index=False)
    print(pad_summary.to_string(index=False))

    elapsed = (time.time() - t0) / 60
    print(f"\n  [seed {seed}] done in {elapsed:.1f} min")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Seeds to run: {SEEDS_TO_RUN}  (seed=42 reuses v2 results)")
    print(f"Epochs per model: {EPOCHS_PER_MODEL}")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # Prepare external cohorts ONCE (image paths only, no loading yet)
    print("\nPreparing external cohort metadata…")
    fitz_df = prepare_fitzpatrick17k()
    pad_df  = prepare_pad_ufes_20()

    # Test cohort (ISIC2018 Test)
    test_df = pd.read_csv(TEST_GT_CSV)
    test_df["age"] = test_df["age"].fillna(test_df["age"].mean())
    test_df["sex"] = test_df["sex"].fillna("unknown")
    test_df["localization"] = test_df["localization"].fillna("unknown")

    print(f"\n  Fitzpatrick17k: N={len(fitz_df)}")
    print(f"  PAD-UFES-20:    N={len(pad_df)}")
    print(f"  ISIC2018 Test:  N={len(test_df)}")

    overall_t0 = time.time()
    for s in SEEDS_TO_RUN:
        run_one_seed(s, fitz_df, pad_df, test_df)
    total_h = (time.time() - overall_t0) / 3600
    print(f"\n{'='*70}\nALL SEEDS DONE in {total_h:.2f}h")
    print(f"Next: run aggregate_seeds.py to compute mean ± SD across seeds.")
