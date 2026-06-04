"""
run_lambda_cf_sweep.py — Appendix C: λ_cf sensitivity sweep
============================================================
Trains the Causal-SCM-age model (M_sp = {age}) under a sweep of L_cf
weights to test §6.3's structural claim that the cohort-dependent
direction is *irreducible* to λ_cf — i.e. that no λ_cf produces a
configuration in which the same regularizer is simultaneously
non-harmful on PAD-UFES-20 and non-helpful on Fitzpatrick17k.

Default protocol (single seed, ~7.5 GPU-h on A100):

    seed     = 42
    M_sp     = {age}
    λ_cf     ∈ {0.01, 0.03, 0.1, 0.3, 1.0}     (5 values, log-spaced)
    epochs   = 30,  batch 32,  lr 1e-4         (same as headline runs)
    output   → history_ISIC2018_scm_lambda_sweep/lam_{value}/

To extend to multi-seed (more statistical power, ~22.5 h for 3 seeds):
edit `SEEDS_TO_RUN` at the top of __main__.

Run:
    CUDA_VISIBLE_DEVICES=0 ./ven/bin/python A_Causal_Audit/run_lambda_cf_sweep.py
"""

import os
import sys
import time
import torch
import torchvision.transforms as T
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

# project root is one level above this script
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ham_isic_pipeline import (
    CachedImageDataset, CachedMetaDataset,
    preprocess_and_cache, preprocess_test_cache,
    evaluate_on_loader, compute_metrics,
)
from external_cohorts import (
    prepare_fitzpatrick17k, prepare_pad_ufes_20, StreamingImageDataset,
)
from causal_scm_model import (
    CausalFactorizedNet, train_causal_factorized_scm,
    sample_counterfactual_metadata,
)


# ============================================================
# Sweep configuration
# ============================================================
SEEDS_TO_RUN = [0, 42, 123]                  # multi-seed sweep for §6.3 leg-3
LAMBDA_CF_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
SPURIOUS_META = ("age",)                     # the cohort-direction configuration
SKIP_IF_DONE = True                          # don't re-train if checkpoint exists

EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 1e-4
LAMBDA_META  = 0.1
LAMBDA_ORTH  = 0.01
LAMBDA_IRM   = 1.0
LAMBDA_SCM   = 1.0
IRM_WARMUP   = 5
CF_WARMUP    = 5

TARGET_SIZE  = (224, 224)

IMAGE_DIR      = os.path.join(ROOT, "data/HAM10000_images_combined_600x450")
SEG_DIR        = os.path.join(ROOT, "data/HAM10000_segmentations_lesion_tschandl")
META_CSV       = os.path.join(ROOT, "data/HAM10000_metadata.csv")
TEST_IMAGE_DIR = os.path.join(ROOT, "data/ISIC2018_Task3_Test_Images")
TEST_GT_CSV    = os.path.join(ROOT, "data/ISIC2018_Task3_Test_GroundTruth.csv")

SAVE_ROOT  = os.path.join(ROOT, "history_ISIC2018_scm_lambda_sweep")
CACHE_ROOT = os.path.join(ROOT, "cache_ISIC2018_seed")     # reuse v6 caches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))


# ============================================================
# Mappings
# ============================================================
ham_df_full = pd.read_csv(META_CSV)
ham_df_full["age"] = ham_df_full["age"].fillna(ham_df_full["age"].mean())
DX_MAP  = {dx: i for i, dx in enumerate(sorted(ham_df_full["dx"].unique()))}
SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
LOC_MAP = {loc: i for i, loc in enumerate(sorted(ham_df_full["localization"].unique()))}
NUM_CLASSES = len(DX_MAP); NUM_SEX = len(SEX_MAP); NUM_LOC = len(LOC_MAP)
NUM_AGE_GROUPS = 3


def set_seeds(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Single-run pipeline (parameterised over (seed, λ_cf))
# ============================================================
def run_one_lambda(seed, lambda_cf, fitz_df, pad_df, test_df):
    tag = f"lam_{lambda_cf:g}_seed_{seed}".replace(".", "p")
    save_dir = os.path.join(SAVE_ROOT, tag)
    os.makedirs(save_dir, exist_ok=True)
    cache_dir = f"{CACHE_ROOT}_{seed}"
    os.makedirs(cache_dir, exist_ok=True)
    print("\n" + "#" * 70)
    print(f"# λ_cf SWEEP  |  seed = {seed}  |  λ_cf = {lambda_cf}")
    print("#" * 70)

    eval_csv = os.path.join(save_dir, "sweep_eval.csv")
    ckpt_path = os.path.join(save_dir, "best_causal_scm.pth")
    if SKIP_IF_DONE and os.path.exists(eval_csv) and os.path.exists(ckpt_path):
        print(f"  ↪ skipping (sweep_eval.csv + checkpoint already exist)")
        return pd.read_csv(eval_csv)

    t0 = time.time()
    set_seeds(seed)

    # ---- HAM split ----
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        ham_df_full, test_size=0.2, stratify=ham_df_full["dx"],
        random_state=seed)

    train_data = preprocess_and_cache(
        train_df, IMAGE_DIR, SEG_DIR, DX_MAP, SEX_MAP, LOC_MAP,
        TARGET_SIZE, os.path.join(cache_dir, "train_data.pt"))
    val_data = preprocess_and_cache(
        val_df, IMAGE_DIR, SEG_DIR, DX_MAP, SEX_MAP, LOC_MAP,
        TARGET_SIZE, os.path.join(cache_dir, "val_data.pt"))

    train_aug = T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_aug = T.Compose([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_loader = DataLoader(
        CachedImageDataset(train_data, transform=train_aug),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(
        CachedImageDataset(val_data, transform=val_aug),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)

    # ---- Train ----
    print(f"\n  training Causal-SCM-age, λ_cf = {lambda_cf} ({EPOCHS} ep)…")
    set_seeds(seed)
    model = CausalFactorizedNet(
        num_classes=NUM_CLASSES, num_sex=NUM_SEX, num_loc=NUM_LOC,
        num_age_groups=NUM_AGE_GROUPS).to(DEVICE)
    actual = next(model.parameters()).device
    print(f"  model on {actual}")

    history, best_acc = train_causal_factorized_scm(
        model, train_loader, val_loader, DEVICE,
        epochs=EPOCHS, lr=LR,
        lambda_meta=LAMBDA_META, lambda_orth=LAMBDA_ORTH,
        lambda_irm=LAMBDA_IRM, lambda_scm=LAMBDA_SCM,
        lambda_cf=lambda_cf,                          # <-- swept
        spurious_meta=SPURIOUS_META,
        irm_warmup_epochs=IRM_WARMUP, cf_warmup_epochs=CF_WARMUP,
        num_envs=NUM_AGE_GROUPS, save_path=ckpt_path)
    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, "history_causal_scm.csv"), index=False)

    # ---- Reload best ----
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # ---- Eval on the two cohorts that matter for §6.3 ----
    def _eval_cohort(df, name):
        img_ds = StreamingImageDataset(df, df["image_path"].values,
                                       target_size=TARGET_SIZE)
        loader = DataLoader(img_ds, batch_size=64, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
        preds, probs, labels = evaluate_on_loader(
            model, loader, DEVICE, "causal_factorized")
        # restricted argmax over cohort-present classes
        FITZ_IDX = [DX_MAP[c] for c in ("mel", "bcc", "akiec")]
        PAD_IDX  = [DX_MAP[c] for c in ("bcc", "akiec", "nv", "bkl", "mel")]
        allowed = FITZ_IDX if name == "Fitzpatrick17k" else PAD_IDX
        mask = np.full(probs.shape[1], -np.inf, dtype=probs.dtype)
        mask[allowed] = 0.0
        preds_r = (probs + mask).argmax(axis=1)
        mel_idx = DX_MAP["mel"]
        m = labels == mel_idx
        return {
            "cohort": name,
            "N": int(len(labels)),
            "accuracy_unrestricted": float((preds   == labels).mean()),
            "accuracy_restricted":   float((preds_r == labels).mean()),
            "sens_mel_unrestricted": float((preds[m]   == labels[m]).mean()) if m.sum() else float("nan"),
            "sens_mel_restricted":   float((preds_r[m] == labels[m]).mean()) if m.sum() else float("nan"),
        }

    print("\n  evaluating Fitzpatrick17k …")
    fitz_row = _eval_cohort(fitz_df, "Fitzpatrick17k")
    print("  evaluating PAD-UFES-20 …")
    pad_row  = _eval_cohort(pad_df,  "PAD-UFES-20")

    summary = pd.DataFrame([
        {"lambda_cf": lambda_cf, "seed": seed, **fitz_row},
        {"lambda_cf": lambda_cf, "seed": seed, **pad_row},
    ])
    summary.to_csv(os.path.join(save_dir, "sweep_eval.csv"), index=False)
    print(summary.to_string(index=False))

    elapsed = (time.time() - t0) / 60
    print(f"\n  done in {elapsed:.1f} min (best val acc {best_acc:.4f})")
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"GPU: {torch.cuda.get_device_name(idx)} "
              f"({torch.cuda.get_device_properties(idx).total_memory / 1024**3:.0f} GB)")
    print(f"\nSeeds       : {SEEDS_TO_RUN}")
    print(f"λ_cf grid   : {LAMBDA_CF_GRID}")
    print(f"M_spurious  : {SPURIOUS_META}")
    print(f"Output      : {SAVE_ROOT}/")
    print(f"Cost        : ~1.5 h per (seed, λ); "
          f"total ~{1.5 * len(SEEDS_TO_RUN) * len(LAMBDA_CF_GRID):.1f} h")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("\nPreparing external cohort metadata…")
    fitz_df = prepare_fitzpatrick17k()
    pad_df  = prepare_pad_ufes_20()
    test_df = pd.read_csv(TEST_GT_CSV)            # kept for parity, unused here
    test_df["age"] = test_df["age"].fillna(test_df["age"].mean())
    test_df["sex"] = test_df["sex"].fillna("unknown")
    test_df["localization"] = test_df["localization"].fillna("unknown")
    print(f"  Fitzpatrick17k: N={len(fitz_df)}  |  PAD-UFES-20: N={len(pad_df)}")

    overall_t0 = time.time()
    for s in SEEDS_TO_RUN:
        for lam in LAMBDA_CF_GRID:
            run_one_lambda(s, lam, fitz_df, pad_df, test_df)

    # Aggregate ALL existing sweep_eval.csv files (including seeds from
    # previous runs that may not be in SEEDS_TO_RUN this time).
    import glob
    all_csvs = sorted(glob.glob(os.path.join(SAVE_ROOT,
                                             "lam_*_seed_*", "sweep_eval.csv")))
    combined = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
    combined.to_csv(os.path.join(SAVE_ROOT, "sweep_all.csv"), index=False)
    print(f"\nAggregated {len(all_csvs)} sweep_eval.csv files into sweep_all.csv")

    print(f"\n{'='*70}\nALL DONE in {(time.time()-overall_t0)/3600:.2f} h")
    print(f"Per-run results: {SAVE_ROOT}/lam_*_seed_*/sweep_eval.csv")
    print(f"Combined:        {SAVE_ROOT}/sweep_all.csv")
