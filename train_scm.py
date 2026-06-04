"""
train_scm.py — Train the feature-space Causal-SCM model for one partition
=========================================================================
Trains the CausalFactorizedNet (the genuinely-fitted feature-space SCM of
causal_scm_model.py) with the counterfactual-invariance loss L_cf enforced
on a chosen *spurious* metadata subset M_sp. The partition is selected on
the command line; everything else (architecture, schedule, loss weights,
10 seeds) is identical across partitions so that the only variable is M_sp.

    python train_scm.py --spurious age      # M_sp = {age}        (headline)
    python train_scm.py --spurious sex      # M_sp = {sex}
    python train_scm.py --spurious sexloc   # M_sp = {sex, loc}
    python train_scm.py --spurious all      # M_sp = {age,sex,loc} (full)
    python train_scm.py --spurious age --seeds 0 42 123   # subset of seeds

The four partitions write to separate result trees so all runs coexist for
the 6-way comparison in eval_restricted_decision.py:

  --spurious all     -> history_ISIC2018_scm/
  --spurious sex     -> history_ISIC2018_scm_partition/
  --spurious sexloc  -> history_ISIC2018_scm_part_sexloc/
  --spurious age     -> history_ISIC2018_scm_part_age/

For each seed s in SEEDS_TO_RUN, <SAVE_ROOT>/seed_{s}/ receives:
     best_causal_scm.pth
     history_causal_scm.csv
     summary_{val,isic2018_test,fitzpatrick17k,pad_ufes_20}.csv
     scm_diagnostics.csv             <-- u_mean ~ 0, u_std ~ 1, cf_shift > 0
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
# Config
# ============================================================
SEEDS_TO_RUN = [0, 1, 2, 3, 7, 11, 31, 42, 99, 123]
EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 1e-4

# Loss weights for train_causal_factorized_scm
LAMBDA_META  = 0.1
LAMBDA_ORTH  = 0.01
LAMBDA_IRM   = 1.0
LAMBDA_SCM   = 1.0      # max-likelihood fit of the structural mechanism f_x
LAMBDA_CF    = 0.1      # counterfactual-invariance regularizer (on M_spurious)
IRM_WARMUP   = 5
CF_WARMUP    = 5        # f_x must be partly fitted before its CFs are useful

# --- Partition selection via CLI -----------------------------------------
# M_sp is the metadata subset made counterfactually invariant; the rest of M
# is left "causal". Each partition writes to its own result tree so the four
# runs coexist for the 6-way comparison in eval_restricted_decision.py.
import argparse
_PARTITIONS = {
    "age":    (("age",),              "history_ISIC2018_scm_part_age"),
    "sex":    (("sex",),              "history_ISIC2018_scm_partition"),
    "sexloc": (("sex", "loc"),        "history_ISIC2018_scm_part_sexloc"),
    "all":    (("age", "sex", "loc"), "history_ISIC2018_scm"),
}
_ap = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--spurious", choices=list(_PARTITIONS), default="age",
                 help="metadata subset M_sp made invariant (default: age)")
_ap.add_argument("--seeds", type=int, nargs="+", default=None,
                 help="override the seed list (default: the 10 paper seeds)")
_args, _ = _ap.parse_known_args()
SPURIOUS_META, _SAVE_ROOT_CLI = _PARTITIONS[_args.spurious]
if _args.seeds is not None:
    SEEDS_TO_RUN = _args.seeds

TARGET_SIZE  = (224, 224)

IMAGE_DIR      = "data/HAM10000_images_combined_600x450"
SEG_DIR        = "data/HAM10000_segmentations_lesion_tschandl"
META_CSV       = "data/HAM10000_metadata.csv"
TEST_IMAGE_DIR = "data/ISIC2018_Task3_Test_Images"
TEST_GT_CSV    = "data/ISIC2018_Task3_Test_GroundTruth.csv"

# Result tree for the selected partition (see _PARTITIONS above).
SAVE_ROOT  = _SAVE_ROOT_CLI
CACHE_ROOT = "cache_ISIC2018_seed"          # reuse the per-seed image caches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))


# ============================================================
# Mappings (identical to train_ISIC_2018_04.py)
# ============================================================
ham_df_full = pd.read_csv(META_CSV)
ham_df_full["age"] = ham_df_full["age"].fillna(ham_df_full["age"].mean())
DX_MAP  = {dx: i for i, dx in enumerate(sorted(ham_df_full["dx"].unique()))}
SEX_MAP = {"male": 0, "female": 1, "unknown": 2}
LOC_MAP = {loc: i for i, loc in enumerate(sorted(ham_df_full["localization"].unique()))}
NUM_CLASSES = len(DX_MAP)
NUM_SEX = len(SEX_MAP)
NUM_LOC = len(LOC_MAP)
NUM_AGE_GROUPS = 3


# ============================================================
# Helpers
# ============================================================
def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def eval_cohort(model, img_loader, name, num_classes):
    """Disease accuracy / sensitivity / specificity for the causal model."""
    print("================ ", DEVICE )
    preds, _, labels = evaluate_on_loader(model, img_loader, DEVICE,
                                          "causal_factorized")
    acc, _, ss = compute_metrics(preds, labels, num_classes)
    return {
        "Model": "Causal-SCM", "Cohort": name, "N": int(len(labels)),
        "Accuracy": acc,
        "Mean Sensitivity": ss["sensitivity"].mean(),
        "Mean Specificity": ss["specificity"].mean(),
    }


def scm_diagnostics(model, img_loader, spurious=SPURIOUS_META):
    """Evidence that the structural mechanism is genuinely fitted.

    A correctly fitted f_x must give:
      * abducted U_x ~ N(0, I)  -> u_mean ~ 0, u_std ~ 1
      * a non-zero counterfactual shift  ||feat(do(M_spurious=m')) - feat|| > 0
    The counterfactual shift is measured under the SAME partial intervention
    used by L_cf in training (re-draw `spurious` components only). These two
    numbers belong in the paper as the SCM sanity check that replaces the
    old Cramer's-V "validation".
    """
    model.eval()
    u_chunks, shift_chunks = [], []
    print("================ ", DEVICE )
    with torch.no_grad():
        for images, labels, ages, sexes, locs, _ in img_loader:
            images = images.to(DEVICE); labels = labels.to(DEVICE)
            ages = ages.to(DEVICE); sexes = sexes.to(DEVICE); locs = locs.to(DEVICE)

            feat = model.encode_feat(images)
            u_x = model.abduct(feat, labels, ages, sexes, locs)
            u_chunks.append(u_x.cpu())

            # do(M_spurious = m') — intervene only on the spurious components
            age_cf, sex_cf, loc_cf = sample_counterfactual_metadata(
                ages, sexes, locs, spurious)
            feat_cf = model.counterfactual_feat(
                feat, labels, ages, sexes, locs, age_cf, sex_cf, loc_cf)
            shift_chunks.append((feat_cf - feat).abs().mean(dim=1).cpu())

    u_all = torch.cat(u_chunks)
    shift = torch.cat(shift_chunks)
    return {
        "u_mean":        float(u_all.mean()),       # target ~ 0
        "u_std":         float(u_all.std()),        # target ~ 1
        "u_abs_mean":    float(u_all.abs().mean()),
        "cf_shift_mean": float(shift.mean()),       # target > 0
        "cf_shift_std":  float(shift.std()),
    }


# ============================================================
# Per-seed pipeline
# ============================================================
def run_one_seed(seed, fitz_df, pad_df, test_df):
    print("\n" + "#" * 70)
    print(f"# CAUSAL-SCM  |  SEED = {seed}")
    print("#" * 70)
    t0 = time.time()

    save_dir = os.path.join(SAVE_ROOT, f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)
    cache_dir = f"{CACHE_ROOT}_{seed}"
    os.makedirs(cache_dir, exist_ok=True)

    set_seeds(seed)

    # ---- Split (matches train_ISIC_2018_04.py) ----
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        ham_df_full, test_size=0.2, stratify=ham_df_full["dx"],
        random_state=seed)

    # ---- Preprocess (reuses the v6 per-seed cache) ----
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

    # ---- Train the Path-2 causal model ----
    print(f"\n  [seed {seed}] training Causal-SCM ({EPOCHS} epochs)…")
    set_seeds(seed)
    model = CausalFactorizedNet(
        num_classes=NUM_CLASSES, num_sex=NUM_SEX, num_loc=NUM_LOC,
        num_age_groups=NUM_AGE_GROUPS).to(DEVICE)
    # Confirm the model really landed on the intended device
    actual = next(model.parameters()).device
    print(f"  [seed {seed}] model on {actual}"
          f"  ({'GPU' if actual.type == 'cuda' else 'CPU'})")

    ckpt_path = os.path.join(save_dir, "best_causal_scm.pth")
    history, best_acc = train_causal_factorized_scm(
        model, train_loader, val_loader, "cuda",
        epochs=EPOCHS, lr=LR,
        lambda_meta=LAMBDA_META, lambda_orth=LAMBDA_ORTH,
        lambda_irm=LAMBDA_IRM, lambda_scm=LAMBDA_SCM, lambda_cf=LAMBDA_CF,
        spurious_meta=SPURIOUS_META,
        irm_warmup_epochs=IRM_WARMUP, cf_warmup_epochs=CF_WARMUP,
        num_envs=NUM_AGE_GROUPS, save_path=ckpt_path)
    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, "history_causal_scm.csv"), index=False)

    # ---- Reload best checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # ---- SCM diagnostics on HAM val ----
    diag = scm_diagnostics(model, val_loader)
    pd.DataFrame([diag]).to_csv(
        os.path.join(save_dir, "scm_diagnostics.csv"), index=False)
    print(f"  [seed {seed}] SCM diagnostics: "
          f"U_x mean={diag['u_mean']:+.3f} std={diag['u_std']:.3f} "
          f"(target 0 / 1)  |  do(M=m') shift={diag['cf_shift_mean']:.4f}")

    # ---- Eval: HAM val ----
    pd.DataFrame([eval_cohort(model, val_loader, "HAM val", NUM_CLASSES)]).to_csv(
        os.path.join(save_dir, "summary_val.csv"), index=False)

    # ---- Eval: ISIC2018 Test ----
    test_data = preprocess_test_cache(
        test_df, TEST_IMAGE_DIR, DX_MAP, SEX_MAP, LOC_MAP,
        TARGET_SIZE, os.path.join(cache_dir, "test_external_data.pt"))
    test_loader = DataLoader(
        CachedImageDataset(test_data, transform=val_aug),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    pd.DataFrame([eval_cohort(model, test_loader, "ISIC2018 Test",
                              NUM_CLASSES)]).to_csv(
        os.path.join(save_dir, "summary_isic2018_test.csv"), index=False)

    # ---- Eval: streaming clinical-photo cohorts ----
    for cohort_df, fname, cohort_name in [
        (fitz_df, "summary_fitzpatrick17k.csv", "Fitzpatrick17k"),
        (pad_df,  "summary_pad_ufes_20.csv",   "PAD-UFES-20"),
    ]:
        img_ds = StreamingImageDataset(cohort_df, cohort_df["image_path"].values,
                                       target_size=TARGET_SIZE)
        img_loader = DataLoader(img_ds, batch_size=64, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
        pd.DataFrame([eval_cohort(model, img_loader, cohort_name,
                                  NUM_CLASSES)]).to_csv(
            os.path.join(save_dir, fname), index=False)

    elapsed = (time.time() - t0) / 60
    print(f"  [seed {seed}] done in {elapsed:.1f} min  (best val acc {best_acc:.4f})")


# ============================================================
# Main
# ============================================================
def print_device_banner():
    """Print clearly whether training runs on GPU or CPU."""
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print("=" * 70)
        print(f">>> Running on GPU  |  cuda:{idx}  {name}  ({total_gb:.0f} GB)")
        print(f"    torch {torch.__version__}  |  CUDA {torch.version.cuda}")
        print("=" * 70)
    else:
        print("=" * 70)
        print(">>> Running on CPU  (no CUDA device visible — training will be slow)")
        print(f"    torch {torch.__version__}")
        print("=" * 70)


if __name__ == "__main__":
    print_device_banner()
    print(f"Seeds: {SEEDS_TO_RUN}  |  Epochs: {EPOCHS}")
    print(f"Loss weights: meta={LAMBDA_META} orth={LAMBDA_ORTH} "
          f"irm={LAMBDA_IRM} scm={LAMBDA_SCM} cf={LAMBDA_CF}")
    _causal_m = tuple(m for m in ("age", "sex", "loc") if m not in SPURIOUS_META)
    print(f"L_cf partition: spurious M = {SPURIOUS_META}  "
          f"(regularised)  |  causal M = {_causal_m}  (left free)")
    print(f"Output dir: {SAVE_ROOT}/")
    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("\nPreparing external cohort metadata…")
    fitz_df = prepare_fitzpatrick17k()
    pad_df  = prepare_pad_ufes_20()
    test_df = pd.read_csv(TEST_GT_CSV)
    test_df["age"] = test_df["age"].fillna(test_df["age"].mean())
    test_df["sex"] = test_df["sex"].fillna("unknown")
    test_df["localization"] = test_df["localization"].fillna("unknown")
    print(f"  Fitzpatrick17k: N={len(fitz_df)}  |  PAD-UFES-20: N={len(pad_df)}"
          f"  |  ISIC2018 Test: N={len(test_df)}")

    overall_t0 = time.time()
    for s in SEEDS_TO_RUN:
        run_one_seed(s, fitz_df, pad_df, test_df)
    print(f"\n{'='*70}\nALL SEEDS DONE in "
          f"{(time.time() - overall_t0) / 3600:.2f}h")
    print(f"Results under {SAVE_ROOT}/seed_*/")
