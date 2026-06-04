# A Causal Audit of Counterfactual-Invariance Regularization in Multimodal Dermatology

Reference implementation for the paper *"There Is No Universal Spurious/Causal
Partition: A Causal Audit of Counterfactual-Invariance Regularization in
Multimodal Dermatology."*

> **Giới thiệu (tóm tắt).** Dự án này cài đặt một **Structural Causal Model
> (SCM) trong không gian đặc trưng** được *khớp thật bằng maximum likelihood*
> (cơ chế location-scale additive-noise), cho phép suy luận phản thực
> *abduction–action–prediction* của Pearl trên metadata nhân khẩu (tuổi, giới,
> vị trí tổn thương). Dùng cơ chế đó, ta **audit** bộ điều chuẩn
> *counterfactual-invariance* `L_cf` qua bốn cách phân vùng tập "spurious"
> `M_sp`, 10 seed, và bốn cohort da liễu. Kết quả chính: cùng một bộ điều chuẩn
> bất biến theo *tuổi* lại **giúp** độ nhạy melanoma trên Fitzpatrick17k nhưng
> **hại** trên PAD-UFES-20 — tức không tồn tại một phân vùng spurious/causal
> phổ quát. Thư mục này chỉ chứa **mã nguồn huấn luyện và đánh giá**
> (không bao gồm mã dựng bảng/hình/bản thảo của bài báo).

---

## What this code does

The pipeline trains a multimodal skin-lesion classifier (EfficientNet-B3
backbone + metadata fusion) coupled to an explicit feature-space SCM, then
audits the counterfactual-invariance regularizer `L_cf` while varying only the
*spurious* metadata subset `M_sp ⊆ {age, sex, loc}`. Everything is trained on a
single dermoscopy source (**HAM10000**) and evaluated cross-cohort on **ISIC
2018 Test**, **Fitzpatrick17k**, and **PAD-UFES-20** across 10 seeds, with
paired statistics (t-test, Wilcoxon, Cohen's d, bootstrap CI).

The genuine SCM (in `causal_scm_model.py`):

```
feat = μ_φ(Y, M) + σ_φ(Y, M) ⊙ U_x          # location-scale additive-noise mechanism
U_x  = (feat − μ_φ(Y, M)) ⊘ σ_φ(Y, M)        # abduction (exact, invertible)
feat_cf(M') = μ_φ(Y, M') + σ_φ(Y, M') ⊙ Û_x  # action + prediction = do(M = M')
```

fit by `L_scm` (NLL of the change-of-variables) and used inside the
counterfactual-stability loss `L_cf` (symmetric KL between factual and
counterfactual disease posteriors). A two-phase per-batch optimizer keeps the
mechanism and the regularizer from collapsing each other.

---

## Repository layout

| File | Role |
|------|------|
| `causal_scm_model.py` | **The contribution.** `CausalFactorizedNet`, the SCM mechanism (`abduct`, `counterfactual_feat`), the losses (`structural_nll_loss`, `counterfactual_invariance_loss`), and the two-phase trainer `train_causal_factorized_scm`. |
| `ham_isic_pipeline.py` | Shared HAM10000 / ISIC 2018 data loading, per-seed caching, and evaluation helpers (`CachedImageDataset`, `preprocess_and_cache`, `evaluate_on_loader`, `compute_metrics`). Heavy utility library from the broader codebase; only a subset of its functions is used here. |
| `external_cohorts.py` | Loaders + label maps for the external clinical-photograph cohorts (`prepare_fitzpatrick17k`, `prepare_pad_ufes_20`, `StreamingImageDataset`). |
| `train_baselines.py` | Trains the two **baselines** per seed → `history_ISIC2018_v6/`: **Image-Only** and **Causal-v2** (the non-counterfactual metadata-permutation control). |
| `train_scm.py` | Trains the **Causal-SCM** model for one partition. `--spurious {age,sex,sexloc,all}` selects `M_sp`; each writes to its own result tree. |
| `eval_restricted_decision.py` | 6-way head-to-head eval (baselines + 4 SCM partitions) on Fitzpatrick17k / PAD-UFES-20 under unrestricted & restricted decision; writes per-seed, aggregated, and all paired contrasts. |
| `run_lambda_cf_sweep.py` | λ_cf sensitivity sweep: retrains Causal-SCM-age over `λ_cf ∈ {0.01,0.03,0.1,0.3,1.0}` to test that the cohort-direction reversal is irreducible to `λ_cf`. |
| `aggregate_seeds.py` | Mean ± SD and paired stats for the baseline runs. |
| `requirements.txt` | Python dependencies. |

> This project contains **training + evaluation source only**. The scripts that
> render the paper's tables/figures and the manuscript itself are intentionally
> excluded; they live one level up in `A_Causal_Audit/`.

---

## Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Experiments were run on a single **NVIDIA A100-80GB** with PyTorch 2.x / CUDA
12+. One seed of any SCM configuration trains in ≈1.5 h (≈15 h for 10 seeds);
the full study is ≈60 GPU-hours of training plus ≈1.5 h of evaluation. A CPU
will run the code but is impractical for full training.

---

## Data layout

All scripts use **paths relative to the current working directory**. Place (or
symlink) the public datasets under `./data/` before running:

```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_combined_600x450/        # HAM10000 images (training source)
├── HAM10000_segmentations_lesion_tschandl/  # lesion masks
├── ISIC2018_Task3_Test_Images/              # ISIC 2018 Task-3 test set
├── ISIC2018_Task3_Test_GroundTruth.csv
├── fitzpatrick17k/
│   ├── fitzpatrick17k.csv
│   └── background removed/                   # <hash>.jpg per image
└── PAD-UFES-20/
    ├── metadata.csv
    └── imgs_part_{1,2,3}/imgs_part_{1,2,3}/  # smartphone photographs
```

The datasets are public; obtain them from their original sources under their
own licenses (HAM10000 — Tschandl et al. 2018; ISIC 2018 Task 3;
Fitzpatrick17k — Groh et al. 2021; PAD-UFES-20 — Pacheco et al. 2020).

If the data already lives elsewhere, symlink it in:

```bash
ln -s /path/to/Causal_ML_2018/data ./data
```

**Outputs** are written to the working directory: per-seed image caches
(`cache_ISIC2018_seed_{s}/`) and result trees (`history_ISIC2018_*/`, including
`…/restricted_decision/*.csv`).

---

## How to run (end-to-end)

Run everything from this project directory. Seeds used in the paper:
`{0, 1, 2, 3, 7, 11, 31, 42, 99, 123}`.

### 1. Baselines (Image-Only + Causal-v2)
```bash
python train_baselines.py
# → history_ISIC2018_v6/seed_{s}/best_image_only.pth, best_causal_factorized.pth
```

### 2. The four Causal-SCM partitions
```bash
python train_scm.py --spurious all      # M_sp = {age,sex,loc}  → history_ISIC2018_scm/
python train_scm.py --spurious sex      # M_sp = {sex}          → history_ISIC2018_scm_partition/
python train_scm.py --spurious sexloc   # M_sp = {sex,loc}      → history_ISIC2018_scm_part_sexloc/
python train_scm.py --spurious age      # M_sp = {age}          → history_ISIC2018_scm_part_age/
# quick test on a subset of seeds:
python train_scm.py --spurious age --seeds 0 42 123
```
Each `history_*/seed_{s}/` gets `best_causal_scm.pth`, `history_causal_scm.csv`,
`summary_*.csv`, and `scm_diagnostics.csv` (Û_x mean≈0, std≈1, cf_shift>0 — the
SCM fit diagnostics of Section 5.1 / Table 4).

### 3. 6-way restricted-decision evaluation (the main results)
```bash
python eval_restricted_decision.py
# → history_ISIC2018_scm_part_age/restricted_decision/{per_seed,aggregated,paired_delta}.csv
```
This produces the numbers behind Tables 1–3 and the cohort-direction finding
(Causal-SCM-age helps melanoma sensitivity on Fitzpatrick17k, harms it on
PAD-UFES-20). Missing checkpoints are skipped with a warning.

### 4. Baseline aggregation (optional)
```bash
python aggregate_seeds.py        # mean ± SD + paired stats for history_ISIC2018_v6
```

### 5. λ_cf sensitivity sweep
```bash
CUDA_VISIBLE_DEVICES=0 python run_lambda_cf_sweep.py
# → history_ISIC2018_scm_lambda_sweep/lam_{value}/  (+ sweep_all.csv)
```

---

## Where the results land

Raw experimental outputs (the inputs to every table/figure in the paper) are
written under the working directory:

| Output | Content |
|---|---|
| `history_ISIC2018_v6/seed_*/summary_*.csv` | per-seed accuracy / sensitivity for the two baselines |
| `history_ISIC2018_scm*/seed_*/summary_*.csv` | per-seed metrics for each SCM partition |
| `history_ISIC2018_scm*/seed_*/scm_diagnostics.csv` | Û_x mean/std and cf_shift (SCM fit diagnostics) |
| `history_ISIC2018_scm*/seed_*/history_causal_scm.csv` | per-epoch training curves |
| `…/restricted_decision/per_seed.csv` | every model × seed × cohort × rule metric |
| `…/restricted_decision/aggregated.csv` | mean ± SD per cell |
| `…/restricted_decision/paired_delta.csv` | every paired contrast (Δ, p, Wilcoxon, d, bootstrap CI) |
| `history_ISIC2018_scm_lambda_sweep/sweep_all.csv` | the λ_cf sweep results |

The figure/table builders that turn these CSVs into the manuscript are **not**
part of this code project (see note above).

---

## Hyper-parameters (defaults, all in the train scripts)

```
epochs 30 · batch 32 · Adam lr 1e-4 · cosine decay · grad-clip 5
λ_meta 0.1 · λ_orth 0.01 · λ_irm 1.0 · λ_scm 1.0 · λ_cf 0.1
IRM/L_cf warm-up: 5 epochs · IRM environments: age strata {<40, 40–60, ≥60}
```

## Notes & caveats

- **Determinism.** No CUDA-determinism flags are pinned; convolutions are
  non-deterministic, so individual runs drift. The headline findings are read
  at the level of cross-seed sign-stability, not bit-reproducibility (see the
  within-seed run-to-run variance discussion in the paper's Appendix C).
- **Melanoma denominators are small** — Fitzpatrick17k `n_mel = 244`,
  PAD-UFES-20 `n_mel = 52`. The PAD effects correspond to ~4 melanoma cases;
  interpret single-cohort magnitudes with that scale in mind and rely on
  cross-seed / cross-λ consistency.
- `ham_isic_pipeline.py` is a large shared utility module carried over verbatim
  from the broader project; only its data-loading and metric helpers are used
  here. It is import-safe (its standalone `__main__` block is never invoked by
  this pipeline).
- The manuscript sources (`paper.md`, `make_pdf.py`, `make_docx.py`) live one
  level up in `A_Causal_Audit/`, not in this code project.
```
