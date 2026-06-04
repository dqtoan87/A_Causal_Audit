"""
eval_restricted_decision_scm.py — restricted-decision head-to-head
==================================================================
Six-way comparison across 10 seeds:

  Image-Only          history_ISIC2018_v6/seed_{s}/best_image_only.pth
  Causal-v2           history_ISIC2018_v6/seed_{s}/best_causal_factorized.pth
  Causal-SCM-all      history_ISIC2018_scm/seed_{s}/best_causal_scm.pth
                      (L_cf invariance to ALL of M = age, sex, loc)
  Causal-SCM-part     history_ISIC2018_scm_partition/seed_{s}/best_causal_scm.pth
                      (L_cf invariance to spurious M = sex only)
  Causal-SCM-sexloc   history_ISIC2018_scm_part_sexloc/seed_{s}/best_causal_scm.pth
                      (L_cf invariance to spurious M = sex + loc; age causal)
  Causal-SCM-age      history_ISIC2018_scm_part_age/seed_{s}/best_causal_scm.pth
                      (L_cf invariance to spurious M = age only;
                       isolation test of the "age destroys sens_mel" claim)

The partition-spurious sweep established:
  * all-M             -> PAD sens_mel 0.17 -> 0.08  (collapsed, 10/10 seeds).
  * sex-only          -> PAD sens_mel 0.13         (recovered ~50%).
  * sex+loc           -> PAD sens_mel 0.115        (no further recovery;
                                                    accuracy DID recover).
  * age-only (this)   -> tests: if PAD sens_mel ~ 0.08, age is empirically
                        the dominant destroyer of melanoma signal.

Any model not yet trained is skipped with a warning; the script still
produces results for whatever checkpoints are available.

and re-evaluates them on Fitzpatrick17k and PAD-UFES-20 under two
decision rules:

  unrestricted   argmax over all 7 HAM classes (operational behaviour).
  restricted     argmax over the cohort-present classes only
                 (Fitz: {mel,bcc,akiec};  PAD: {bcc,akiec,nv,bkl,mel}).

Two metrics are reported per cohort x rule x model:

  accuracy       overall accuracy.
  sens_mel       melanoma sensitivity -- the clinically decisive metric
                 (a false-negative melanoma is the costly error). The
                 v6 restricted-decision analysis showed Causal-v2 LOSES
                 aggregate accuracy on Fitzpatrick but GAINS melanoma
                 sensitivity; this script checks whether Causal-SCM
                 repeats or amplifies that pattern.

Paired contrasts (per cohort x rule x metric):
  Causal-SCM-age    - Image-Only        (does age-only invariance hurt?)
  Causal-SCM-age    - Causal-SCM-all    (does age explain most all-M harm?)
  Causal-SCM-age    - Causal-SCM-part   (does age hurt more than sex?)
  Causal-SCM-sexloc - Causal-SCM-part   (does adding loc to spurious help?)
  Causal-SCM-sexloc - Image-Only
  Causal-SCM-part   - Causal-SCM-all
  Causal-SCM-part   - Image-Only
  Causal-SCM-part   - Causal-v2
  Causal-SCM-all    - Image-Only        (reproduce the all-M negative result)
  Causal-v2         - Image-Only        (reproduce v6 partC)

Output (NEW directory, preserves earlier result dirs):
  history_ISIC2018_scm_part_age/restricted_decision/per_seed.csv
  history_ISIC2018_scm_part_age/restricted_decision/aggregated.csv
  history_ISIC2018_scm_part_age/restricted_decision/paired_delta.csv

Reads existing checkpoints only; no training. ~1.5-2 h on GPU (I/O dominated).
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ham_isic_pipeline import (
    CausalFactorizedNet as CausalV2Net,
    ImageOnlyNet,
    evaluate_on_loader,
)
from causal_scm_model import CausalFactorizedNet as CausalSCMNet
from external_cohorts import (
    prepare_fitzpatrick17k, prepare_pad_ufes_20,
    StreamingImageDataset,
    DX_MAP, NUM_CLASSES, NUM_SEX, NUM_LOC,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 7, 11, 31, 42, 99, 123]

V6_ROOT         = "history_ISIC2018_v6"
SCM_ALL_ROOT    = "history_ISIC2018_scm"             # all-M invariance
SCM_PART_ROOT   = "history_ISIC2018_scm_partition"   # sex-only invariance
SCM_SEXLOC_ROOT = "history_ISIC2018_scm_part_sexloc" # sex+loc invariance
SCM_AGE_ROOT    = "history_ISIC2018_scm_part_age"    # age-only invariance
SAVE_DIR        = os.path.join(SCM_AGE_ROOT, "restricted_decision")
os.makedirs(SAVE_DIR, exist_ok=True)

# Cohort-present classes (HAM 7-class sorted: akiec,bcc,bkl,df,mel,nv,vasc)
FITZ_CLASSES = ["mel", "bcc", "akiec"]
PAD_CLASSES  = ["bcc", "akiec", "nv", "bkl", "mel"]
FITZ_IDX = [DX_MAP[c] for c in FITZ_CLASSES]
PAD_IDX  = [DX_MAP[c] for c in PAD_CLASSES]

MODEL_ORDER = ["Image-Only", "Causal-v2",
               "Causal-SCM-all", "Causal-SCM-part",
               "Causal-SCM-sexloc", "Causal-SCM-age"]
# Per-cohort x rule x metric contrasts: (model_A, model_B) -> tests A - B > 0
CONTRASTS = [
    # --- this run's decisive contrasts ---
    ("Causal-SCM-age",    "Image-Only"),        # age invariance hurts vs baseline?
    ("Causal-SCM-age",    "Causal-SCM-all"),    # age alone ~ all-M harm?
    ("Causal-SCM-age",    "Causal-SCM-part"),   # age worse than sex?
    # --- prior decisive contrasts (kept for cross-reference) ---
    ("Causal-SCM-sexloc", "Causal-SCM-part"),
    ("Causal-SCM-sexloc", "Image-Only"),
    ("Causal-SCM-part",   "Causal-SCM-all"),
    ("Causal-SCM-part",   "Image-Only"),
    ("Causal-SCM-part",   "Causal-v2"),
    ("Causal-SCM-all",    "Image-Only"),
    ("Causal-v2",         "Image-Only"),
]
METRICS = ["accuracy", "sens_mel"]


# ============================================================
# Model loading
# ============================================================
def load_models_for_seed(seed):
    """Return {display_name: (model, eval_type)} for one seed.

    Missing checkpoints are skipped with a warning so a partial run
    still produces results for the models that are available.
    """
    out = {}

    specs = [
        ("Image-Only",         "image_only",
         os.path.join(V6_ROOT,         f"seed_{seed}", "best_image_only.pth"),
         lambda: ImageOnlyNet(NUM_CLASSES)),
        ("Causal-v2",          "causal_factorized",
         os.path.join(V6_ROOT,         f"seed_{seed}", "best_causal_factorized.pth"),
         lambda: CausalV2Net(NUM_CLASSES, NUM_SEX, NUM_LOC)),
        ("Causal-SCM-all",     "causal_factorized",
         os.path.join(SCM_ALL_ROOT,    f"seed_{seed}", "best_causal_scm.pth"),
         lambda: CausalSCMNet(NUM_CLASSES, NUM_SEX, NUM_LOC)),
        ("Causal-SCM-part",    "causal_factorized",
         os.path.join(SCM_PART_ROOT,   f"seed_{seed}", "best_causal_scm.pth"),
         lambda: CausalSCMNet(NUM_CLASSES, NUM_SEX, NUM_LOC)),
        ("Causal-SCM-sexloc",  "causal_factorized",
         os.path.join(SCM_SEXLOC_ROOT, f"seed_{seed}", "best_causal_scm.pth"),
         lambda: CausalSCMNet(NUM_CLASSES, NUM_SEX, NUM_LOC)),
        ("Causal-SCM-age",     "causal_factorized",
         os.path.join(SCM_AGE_ROOT,    f"seed_{seed}", "best_causal_scm.pth"),
         lambda: CausalSCMNet(NUM_CLASSES, NUM_SEX, NUM_LOC)),
    ]
    for name, mtype, ckpt, ctor in specs:
        if not os.path.exists(ckpt):
            print(f"  ! seed {seed}: missing {ckpt}")
            continue
        m = ctor().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.eval()
        out[name] = (m, mtype)
    return out


# ============================================================
# Decision rules
# ============================================================
def restricted_argmax(probs, allowed_idx):
    """argmax over allowed_idx only. probs is (N, 7)."""
    mask = np.full(probs.shape[1], -np.inf, dtype=probs.dtype)
    mask[allowed_idx] = 0.0
    return (probs + mask).argmax(axis=1)


def per_class_sensitivity(preds, labels, allowed_idx, allowed_classes):
    """Sensitivity (recall) for each cohort-present class."""
    sens = {}
    for ci, cn in zip(allowed_idx, allowed_classes):
        m = labels == ci
        sens[cn] = float((preds[m] == labels[m]).mean()) if m.sum() else float("nan")
    return sens


# ============================================================
# Per-seed evaluation
# ============================================================
def eval_all(fitz_df, pad_df):
    rows = []
    cohorts = [
        ("Fitzpatrick17k", fitz_df, FITZ_IDX, FITZ_CLASSES),
        ("PAD-UFES-20",    pad_df,  PAD_IDX,  PAD_CLASSES),
    ]
    for seed in SEEDS:
        print(f"\n[seed {seed}] loading checkpoints …", flush=True)
        models = load_models_for_seed(seed)
        if not models:
            continue

        for cohort_name, df, allowed_idx, allowed_classes in cohorts:
            print(f"  [{cohort_name}] N={len(df)} — "
                  f"evaluating {len(models)} models …", flush=True)
            img_ds = StreamingImageDataset(
                df, df["image_path"].values, target_size=(224, 224))
            loader = DataLoader(img_ds, batch_size=64, shuffle=False,
                                num_workers=0, pin_memory=True)

            for mname, (mdl, mtype) in models.items():
                preds_u, probs, labels = evaluate_on_loader(
                    mdl, loader, DEVICE, mtype)
                preds_r = restricted_argmax(probs, allowed_idx)

                for rule, preds in [("unrestricted", preds_u),
                                    ("restricted",   preds_r)]:
                    acc = float((preds == labels).mean())
                    sens = per_class_sensitivity(
                        preds, labels, allowed_idx, allowed_classes)
                    rows.append({
                        "cohort": cohort_name,
                        "decision_rule": rule,
                        "seed": seed,
                        "model": mname,
                        "N": int(len(labels)),
                        "accuracy": acc,
                        **{f"sens_{cn}": sens[cn] for cn in allowed_classes},
                    })
    return pd.DataFrame(rows)


# ============================================================
# Aggregation
# ============================================================
def aggregate(per_seed):
    """Mean +/- SD of accuracy and sens_mel per cohort x rule x model."""
    out = []
    for (cohort, rule, model), sub in per_seed.groupby(
            ["cohort", "decision_rule", "model"]):
        row = {"cohort": cohort, "decision_rule": rule, "model": model,
               "n_seeds": len(sub)}
        for metric in METRICS:
            vals = sub[metric].dropna().values
            row[f"{metric}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            row[f"{metric}_sd"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        out.append(row)
    df = pd.DataFrame(out)
    # stable ordering
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    return df.sort_values(["cohort", "decision_rule", "model"]).reset_index(drop=True)


def paired_delta(per_seed):
    """Paired contrasts per cohort x rule x metric x (model_A - model_B).

    One-sided alternative Delta > 0 (= 'model_A better than model_B').
    Reports paired-t p, Wilcoxon p, paired Cohen's d, bootstrap 95% CI.
    """
    rng = np.random.default_rng(20260522)
    rows = []
    for (cohort, rule), sub in per_seed.groupby(["cohort", "decision_rule"]):
        for metric in METRICS:
            by_model = {
                m: g.set_index("seed")[metric]
                for m, g in sub.groupby("model")
            }
            for a, b in CONTRASTS:
                if a not in by_model or b not in by_model:
                    continue
                seeds = sorted(set(by_model[a].dropna().index)
                               & set(by_model[b].dropna().index))
                if len(seeds) < 2:
                    continue
                va = np.array([by_model[a][s] for s in seeds])
                vb = np.array([by_model[b][s] for s in seeds])
                deltas = va - vb

                t_stat, t_p_two = stats.ttest_rel(va, vb)
                p_one = (t_p_two / 2 if t_stat > 0 else 1 - t_p_two / 2)
                try:
                    w_p = float(stats.wilcoxon(
                        deltas, alternative="greater").pvalue)
                except ValueError:                       # all-zero deltas
                    w_p = float("nan")
                sd = deltas.std(ddof=1)
                cohen_d = float(deltas.mean() / (sd + 1e-12))
                boot = np.array([
                    rng.choice(deltas, size=len(deltas), replace=True).mean()
                    for _ in range(10000)])
                ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

                rows.append({
                    "cohort": cohort, "decision_rule": rule,
                    "metric": metric, "contrast": f"{a} - {b}",
                    "n_seeds": len(seeds),
                    "delta_mean": float(deltas.mean()),
                    "delta_sd": float(sd),
                    "t_p_one_sided": float(p_one),
                    "wilcoxon_p_one_sided": w_p,
                    "cohen_d_paired": cohen_d,
                    "boot_ci_lo": float(ci_lo),
                    "boot_ci_hi": float(ci_hi),
                })
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"DX_MAP = {DX_MAP}")
    print(f"Fitz allowed idx = {FITZ_IDX} {FITZ_CLASSES}")
    print(f"PAD  allowed idx = {PAD_IDX} {PAD_CLASSES}\n")

    print("Preparing Fitzpatrick17k …")
    fitz_df = prepare_fitzpatrick17k()
    print(f"  Fitz N = {len(fitz_df)}")
    print("Preparing PAD-UFES-20 …")
    pad_df = prepare_pad_ufes_20()
    print(f"  PAD  N = {len(pad_df)}")

    per_seed = eval_all(fitz_df, pad_df)
    per_seed.to_csv(os.path.join(SAVE_DIR, "per_seed.csv"), index=False)

    agg = aggregate(per_seed)
    agg.to_csv(os.path.join(SAVE_DIR, "aggregated.csv"), index=False)

    delta = paired_delta(per_seed)
    delta.to_csv(os.path.join(SAVE_DIR, "paired_delta.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("\n" + "=" * 78)
    print("AGGREGATED  (mean +/- SD over seeds)")
    print("=" * 78)
    show = agg.copy()
    for metric in METRICS:
        show[metric] = show.apply(
            lambda r, m=metric: f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_sd']:.4f}",
            axis=1)
    print(show[["cohort", "decision_rule", "model",
                "n_seeds", "accuracy", "sens_mel"]].to_string(index=False))

    print("\n" + "=" * 78)
    print("PAIRED CONTRASTS  (one-sided: delta > 0 means model_A better)")
    print("=" * 78)
    print(delta.to_string(index=False))

    # ---- Headlines: the two decisive cohort x rule comparisons ----
    def _print_headline(cohort, rule):
        print("\n" + "=" * 78)
        print(f"HEADLINE — {cohort}, {rule} decision")
        print("=" * 78)
        sub = delta[(delta.cohort == cohort) & (delta.decision_rule == rule)]
        for _, r in sub.iterrows():
            verdict = ("A better" if r.delta_mean > 0 and r.t_p_one_sided < 0.05
                       else "B better" if r.delta_mean < 0 and r.t_p_one_sided > 0.95
                       else "n.s.")
            print(f"  {r.metric:10s} | {r.contrast:32s} | "
                  f"Δ={r.delta_mean:+.4f} ± {r.delta_sd:.4f}  "
                  f"p1={r.t_p_one_sided:.4f}  d={r.cohen_d_paired:+.2f}  "
                  f"CI[{r.boot_ci_lo:+.4f},{r.boot_ci_hi:+.4f}]  -> {verdict}")

    # PAD melanoma sensitivity is the decisive metric -- this is where
    # the all-M run halved sens_mel (0.17 -> 0.08). Print it first.
    _print_headline("PAD-UFES-20",    "restricted")
    _print_headline("Fitzpatrick17k", "restricted")

    print(f"\nOutputs written to {SAVE_DIR}/")
