"""
causal_scm_model.py — Path-2 redesign: a genuinely causal CausalFactorizedNet
=============================================================================

Why this file exists
--------------------
The PLOS ONE reviewer (concern #2) and the causal-inference critique make the
same point: the v1/v2 "counterfactual" regularizer is NOT a counterfactual. It
permuted the metadata vector `m` within a batch while keeping the image feature
`feat` FIXED, then penalised the change in the disease prediction. Under the
paper's own SCM (m -> x), an intervention on `m` must propagate through the
structural mechanism f_x that generates the image. Holding `feat` fixed
therefore enforces a *marginal-independence* constraint, not do(M = m').

This module replaces that with a genuine counterfactual. It introduces an
explicit Structural Causal Model in image-FEATURE space and computes
counterfactuals by Pearl's three-step procedure (abduction -> action ->
prediction; Pearl 2009, Causality, Ch. 7).

The SCM (feature space)
-----------------------
    Y            disease label             (exogenous cause of lesion appearance)
    M=(a,s,l)    age / sex / localization   (exogenous metadata)
    U_x          exogenous image noise      (everything else: lesion-intrinsic)
    feat = f_x(Y, M, U_x)                   structural mechanism for the feature

f_x is a *location-scale additive-noise model* (a nonlinear ANM, Hoyer et al.
2009, in its location-scale form -- equivalently a single conditional
normalising-flow layer, Khemakhem et al. 2021):

    feat = mu_phi(Y, M)  +  sigma_phi(Y, M) (.) U_x          (generate)
    U_x  = (feat - mu_phi(Y, M)) / sigma_phi(Y, M)           (abduct / invert)

Because f_x is invertible in U_x given (Y, M), the structural noise of an
observed individual is *identified* by abduction, and the counterfactual
"what would this lesion's feature have been had the patient metadata been m'
instead of m" is well defined and exact:

    Step 1 (abduction):  U_x  = (feat - mu(Y, m )) / sigma(Y, m )
    Step 2 (action):     replace m by m'   -- the only edited structural eq.
    Step 3 (prediction): feat_cf = mu(Y, m') + sigma(Y, m') (.) U_x

`CausalFactorizedNet.counterfactual_feat(...)` performs all three steps.

Two training objectives make the SCM real
------------------------------------------
  * L_scm (`structural_nll_loss`) -- maximum-likelihood fit of f_x. With a
    standard-Normal prior on U_x, the change-of-variables formula gives the
    per-sample NLL  0.5||U_x||^2 + sum_d log sigma_d. Minimising it makes
    mu -> E[feat|Y,M], sigma -> Std[feat|Y,M], so the abducted U_x is a true
    ~N(0,I) exogenous noise. A round-trip RECONSTRUCTION loss cannot fit f_x:
    the map is invertible by construction, so its reconstruction error is
    identically zero.
  * L_cf (`counterfactual_invariance_loss`) -- penalises the change in the
    disease posterior between `feat` and `feat_cf = feat(do(M=m'))`. Because
    feat_cf is produced by f_x, this is a counterfactual-stability constraint
    Y(do(M=m')) = Y(do(M=m)) (Veitch et al. 2021), not marginal independence.

`train_causal_factorized_scm` trains both jointly. Two design rules keep the
objectives from collapsing each other:
  (1) L_scm is fit to `feat.detach()` -- the NLL must not back-propagate into
      the backbone, or the backbone would degrade `feat` to be trivially
      modelled by f_x.
  (2) The mechanism f_x is held FIXED while L_cf is measured -- otherwise the
      optimiser would make f_x map every m' back to `feat` (feat_cf == feat)
      and drive L_cf to zero without changing the classifier. The SCM owns its
      embeddings (it does not share them with the classifier), so freezing
      `model.scm` freezes the mechanism completely.

Assumptions to state explicitly in the paper (Section 2)
--------------------------------------------------------
  * f_x acts in EfficientNet feature space, not pixel space. A pixel-space
    counterfactual (diffusion / deep-SCM image generator) is the stronger
    version and is named as future work.
  * Location-scale additive noise: a stated, identifiable model class
    (Hoyer 2009; Khemakhem 2021), not an unstated convenience.
  * Y is a parent of `feat`. At training time Y is observed, so the
    counterfactual regularizer is a training-time object only. At inference Y
    is unknown, so the disease head reads `feat` via proj_c, never U_x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torchvision import models

try:
    from tqdm import tqdm
except Exception:                                   # tqdm is optional
    def tqdm(it, *a, **k):
        return it


# ============================================================
# Structural mechanism  f_x : (Y, M, U_x) -> feat
# ============================================================
class FeatureSCM(nn.Module):
    """Location-scale additive-noise structural mechanism for the image feature.

        feat = mu(Y, M) + sigma(Y, M) (.) U_x

    Invertible in U_x given (Y, M), so abduction is exact. The mechanism owns
    its OWN embeddings of (Y, age, sex, loc); it shares nothing with the
    classifier, so `freeze(scm)` freezes the whole mechanism in one call.
    """

    SIGMA_FLOOR = 1e-3   # keeps the scale strictly positive -> f_x invertible

    def __init__(self, feat_dim, num_classes, num_sex, num_loc,
                 emb_dim=32, cond_hidden=128, hidden=768):
        super().__init__()
        # --- mechanism-private embeddings of the structural parents ---
        self.age_fc  = nn.Sequential(nn.Linear(1, emb_dim), nn.ReLU())
        self.sex_emb = nn.Embedding(num_sex, emb_dim)
        self.loc_emb = nn.Embedding(num_loc, emb_dim)
        self.y_emb   = nn.Embedding(num_classes, emb_dim)
        self.cond_fc = nn.Sequential(
            nn.Linear(4 * emb_dim, cond_hidden), nn.ReLU())

        # --- shared trunk -> (location, log-scale) ---
        self.trunk = nn.Sequential(
            nn.Linear(cond_hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_mu    = nn.Linear(hidden, feat_dim)
        self.to_log_s = nn.Linear(hidden, feat_dim)
        # init log-scale ~ 0  ->  the model starts near feat = mu + U_x
        nn.init.zeros_(self.to_log_s.weight)
        nn.init.zeros_(self.to_log_s.bias)

    def _cond(self, y, age, sex, loc):
        e = torch.cat([self.age_fc(age.unsqueeze(-1)),
                       self.sex_emb(sex), self.loc_emb(loc),
                       self.y_emb(y)], dim=1)
        return self.cond_fc(e)

    def _loc_scale(self, y, age, sex, loc):
        h = self.trunk(self._cond(y, age, sex, loc))
        mu = self.to_mu(h)
        sigma = Fn.softplus(self.to_log_s(h)) + self.SIGMA_FLOOR
        return mu, sigma

    def abduct(self, feat, y, age, sex, loc):
        """Step 1 -- recover the exogenous noise U_x of an observed feature."""
        mu, sigma = self._loc_scale(y, age, sex, loc)
        return (feat - mu) / sigma

    def generate(self, u_x, y, age, sex, loc):
        """Step 3 -- generate the feature from noise under (possibly new) M."""
        mu, sigma = self._loc_scale(y, age, sex, loc)
        return mu + sigma * u_x

    def nll(self, feat, y, age, sex, loc):
        """Per-sample negative log-likelihood of `feat` under the mechanism.

        feat = mu + sigma (.) U_x  is invertible U_x <-> feat given (Y, M).
        With a standard-Normal prior on U_x, the change-of-variables formula
        gives, up to an additive constant,

            -log p(feat | Y, M) = 0.5 ||U_x||^2 + sum_d log sigma_d.

        Minimising it fits mu -> E[feat|Y,M] and sigma -> Std[feat|Y,M].
        """
        mu, sigma = self._loc_scale(y, age, sex, loc)
        u_x = (feat - mu) / sigma
        return 0.5 * u_x.pow(2).sum(dim=1) + torch.log(sigma).sum(dim=1)


# ============================================================
# CausalFactorizedNet (Path-2)
# ============================================================
class CausalFactorizedNet(nn.Module):
    """Multimodal skin-lesion classifier with an explicit feature-space SCM.

    Forward (inference, Y unknown):
        image -> backbone -> feat
        feat               -> proj_c -> Z_c -> disease_head
        [feat || meta_emb] -> proj_s -> Z_s -> sex / loc / age heads

    Causal machinery (training, Y observed):
        self.scm (FeatureSCM) provides abduction / generation, so a genuine
        counterfactual  feat_cf = feat(do(M=m'))  is available to regularize
        the disease pathway towards counterfactual invariance.
    """

    FEAT_DIM = 1536   # EfficientNet-B3

    def __init__(self, num_classes=7, num_sex=3, num_loc=15,
                 num_age_groups=3, z_dim=512, meta_emb_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.num_sex = num_sex
        self.num_loc = num_loc
        feat_dim = self.FEAT_DIM

        # ---- Image backbone ----
        self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()

        # ---- Classifier-side metadata embedding (feeds proj_s only) ----
        self.age_fc  = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.sex_emb = nn.Embedding(num_sex, 32)
        self.loc_emb = nn.Embedding(num_loc, 32)
        self.meta_fc = nn.Sequential(nn.Linear(96, meta_emb_dim), nn.ReLU())

        # ---- Structural mechanism f_x (owns its own embeddings) ----
        self.scm = FeatureSCM(feat_dim, num_classes, num_sex, num_loc)

        # ---- Factorized projections ----
        # proj_c: image feature ONLY -- no metadata path blocks the trivial leak
        self.proj_c = nn.Sequential(
            nn.Linear(feat_dim, 768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, z_dim),
        )
        # proj_s: image + metadata -- allowed to encode the shortcut
        self.proj_s = nn.Sequential(
            nn.Linear(feat_dim + meta_emb_dim, 768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, z_dim),
        )

        # ---- Heads ----
        self.disease_head = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        self.sex_head = nn.Linear(z_dim, num_sex)
        self.loc_head = nn.Linear(z_dim, num_loc)
        self.age_group_head = nn.Linear(z_dim, num_age_groups)

    # --------------------------------------------------------
    # Parameter groups -- the SCM is optimised, but separately
    # --------------------------------------------------------
    def scm_parameters(self):
        return self.scm.parameters()

    def non_scm_parameters(self):
        scm_ids = {id(p) for p in self.scm.parameters()}
        return [p for p in self.parameters() if id(p) not in scm_ids]

    def set_scm_requires_grad(self, flag):
        for p in self.scm.parameters():
            p.requires_grad_(flag)

    # --------------------------------------------------------
    # Backbone + classifier-side metadata embedding
    # --------------------------------------------------------
    def encode_feat(self, x):
        return self.backbone(x)

    def embed_meta(self, age, sex, loc):
        age_e = self.age_fc(age.unsqueeze(-1))
        sex_e = self.sex_emb(sex)
        loc_e = self.loc_emb(loc)
        return self.meta_fc(torch.cat([age_e, sex_e, loc_e], dim=1))

    # --------------------------------------------------------
    # Factorized decode (inference path -- Y not required)
    # --------------------------------------------------------
    def decode(self, feat, age, sex, loc):
        """feat + metadata -> all task outputs.

        Returns: disease_logits, sex_logits, loc_logits, age_logits, z_c, z_s
        """
        meta = self.embed_meta(age, sex, loc)
        z_c = self.proj_c(feat)
        z_s = self.proj_s(torch.cat([feat, meta], dim=1))

        disease_logits = self.disease_head(z_c)
        sex_logits = self.sex_head(z_s)
        loc_logits = self.loc_head(z_s)
        age_logits = self.age_group_head(z_s)
        return disease_logits, sex_logits, loc_logits, age_logits, z_c, z_s

    def forward(self, x, age, sex, loc):
        return self.decode(self.encode_feat(x), age, sex, loc)

    # --------------------------------------------------------
    # Causal machinery -- abduction / action / prediction
    # --------------------------------------------------------
    def abduct(self, feat, y, age, sex, loc):
        """Step 1. Recover the exogenous noise U_x (needs the observed Y)."""
        return self.scm.abduct(feat, y, age, sex, loc)

    def structural_feat(self, u_x, y, age, sex, loc):
        """Step 3. Generate feat from noise under metadata (age, sex, loc)."""
        return self.scm.generate(u_x, y, age, sex, loc)

    def counterfactual_feat(self, feat, y, age, sex, loc,
                            age_cf, sex_cf, loc_cf):
        """Full abduction -> action -> prediction.

        Returns feat_cf: the feature the SAME lesion (same Y, same U_x) would
        have produced had the patient metadata been (age_cf, sex_cf, loc_cf).
        This is do(M = m') propagated through f_x -- a genuine counterfactual.
        """
        u_x = self.scm.abduct(feat, y, age, sex, loc)            # step 1
        return self.scm.generate(u_x, y, age_cf, sex_cf, loc_cf)  # steps 2-3

    def structural_nll_loss(self, feat, y, age, sex, loc):
        """L_scm -- mean per-sample NLL; the maximum-likelihood fit of f_x.

        Pass `feat.detach()` from the training loop: the NLL must not
        back-propagate into the backbone.
        """
        return self.scm.nll(feat, y, age, sex, loc).mean()


# ============================================================
# Counterfactual-invariance regularizer  (L_cf)
# ============================================================
SPURIOUS_DEFAULT = ("sex",)   # see counterfactual_invariance_loss docstring


def sample_counterfactual_metadata(age, sex, loc, spurious=SPURIOUS_DEFAULT,
                                    generator=None):
    """Draw an alternative metadata vector m' that intervenes ONLY on the
    spurious components of M.

    `spurious` is a subset of {"age", "sex", "loc"}. A component listed in
    `spurious` is re-drawn (batch permutation -> do(. ~ empirical marginal));
    a component NOT listed is held at its observed value. This realises a
    *partial* intervention do(M_spurious = m') that leaves the genuine
    risk-factor components of M untouched.

    Edge cases that make this a clean ablation axis:
      spurious = ("age","sex","loc")  -> intervene on all M (old behaviour)
      spurious = ("sex",)             -> intervene on sex only (default)
      spurious = ()                   -> m' == m, L_cf == 0 (no-CF ablation)
    """
    perm = torch.randperm(age.size(0), device=age.device, generator=generator)
    age_cf = age[perm] if "age" in spurious else age
    sex_cf = sex[perm] if "sex" in spurious else sex
    loc_cf = loc[perm] if "loc" in spurious else loc
    return age_cf, sex_cf, loc_cf


def counterfactual_invariance_loss(model, feat, y, age, sex, loc,
                                   spurious=SPURIOUS_DEFAULT,
                                   disease_logits=None, generator=None):
    """L_cf -- penalise the shift in the disease posterior under
    do(M_spurious = m'), the intervention on the SPURIOUS part of M only.

    Steps:
      1. Sample m' from the empirical marginal, re-drawing ONLY the
         components in `spurious` and holding the rest at their observed
         value (see `sample_counterfactual_metadata`).
      2. feat_cf = model.counterfactual_feat(...)   (propagates through f_x).
      3. Penalise the symmetric KL between p(Y | feat) and p(Y | feat_cf).

    Why a PARTITION of M (the Path-2.1 refinement)
    ----------------------------------------------
    Enforcing invariance to ALL of M was found to be harmful: on
    PAD-UFES-20 it roughly halved melanoma sensitivity (0.17 -> 0.08),
    because lesion site and patient age are GENUINE, transportable
    melanoma risk factors -- a model should use them. Counterfactual
    invariance is only appropriate for the *confounding* part of M.

    Default partition `spurious = ("sex",)`:
      * sex        -- SPURIOUS. Holding the lesion image fixed, recorded
                      sex should not change the diagnosis; sex-linked
                      incidence differences are population-level and
                      cohort-dependent (not transportable). -> regularise.
      * localization -- CAUSAL. Site genuinely drives lesion biology
                      (acral lentiginous melanoma; lentigo maligna of the
                      face). -> NOT regularised.
      * age        -- CAUSAL. A genuine, transportable risk factor
                      (melanoma incidence rises monotonically with age).
                      -> NOT regularised.
    The partition is a declared modelling assumption; justify it in the
    paper with epidemiology and, ideally, conditional-independence tests.

    This is the counterfactual-stability constraint Y(do(M_sp=m')) =
    Y(do(M_sp=m)), NOT marginal independence: feat -> feat_cf is generated
    by the SCM. The caller must hold f_x fixed (see train_causal_factorized_scm).

    `disease_logits` (factual) may be passed in to avoid recomputation.
    """
    age_cf, sex_cf, loc_cf = sample_counterfactual_metadata(
        age, sex, loc, spurious, generator)

    feat_cf = model.counterfactual_feat(feat, y, age, sex, loc,
                                        age_cf, sex_cf, loc_cf)

    if disease_logits is None:
        disease_logits = model.decode(feat, age, sex, loc)[0]
    logits_cf = model.decode(feat_cf, age_cf, sex_cf, loc_cf)[0]

    log_p = Fn.log_softmax(disease_logits, dim=1)
    log_q = Fn.log_softmax(logits_cf,      dim=1)
    p, q = log_p.exp(), log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(1)
    kl_qp = (q * (log_q - log_p)).sum(1)
    return 0.5 * (kl_pq + kl_qp).mean()


# ============================================================
# IRM penalty (IRMv1, Arjovsky et al. 2019) -- self-contained copy
# ============================================================
def irm_penalty(disease_logits, labels, env_ids, num_envs=3):
    """IRMv1 penalty: sum_e || grad_w R_e(w) |_{w=1} ||^2."""
    penalty = 0.0
    for e in range(num_envs):
        mask = (env_ids == e)
        if mask.sum() < 2:
            continue
        scale = torch.ones(1, device=disease_logits.device, requires_grad=True)
        loss_e = Fn.cross_entropy(disease_logits[mask] * scale, labels[mask])
        grad = torch.autograd.grad(loss_e, scale, create_graph=True)[0]
        penalty = penalty + grad.pow(2).sum()
    if isinstance(penalty, float):
        return torch.tensor(0.0, device=disease_logits.device)
    return penalty


# ============================================================
# Training loop  --  L_dis + L_meta + L_orth + L_irm + L_scm + L_cf
# ============================================================
def train_causal_factorized_scm(model, train_loader, val_loader, device,
                                 epochs, lr,
                                 lambda_meta=0.1, lambda_orth=0.01,
                                 lambda_irm=1.0, lambda_scm=1.0,
                                 lambda_cf=0.1,
                                 spurious_meta=SPURIOUS_DEFAULT,
                                 irm_warmup_epochs=5, cf_warmup_epochs=5,
                                 num_envs=3, grad_clip=5.0,
                                 save_path="best_causal_scm.pth", verbose=True):
    """Train the Path-2 CausalFactorizedNet.

    Loss = L_dis
         + lambda_meta * L_meta            (metadata heads off Z_s)
         + lambda_orth * L_orth            (Z_c _|_ Z_s, vector orthogonality)
         + lambda_irm  * L_irm             (IRMv1 across age environments)
         + lambda_scm  * L_scm             (max-likelihood fit of f_x)
         + lambda_cf   * L_cf              (counterfactual invariance under
                                            do(M_spurious=m'); `spurious_meta`
                                            selects which M components are
                                            regularised -- see
                                            counterfactual_invariance_loss)

    Each batch is optimised in two back-prop passes sharing one optimiser /
    one .step():

      Pass A -- fit the mechanism.  L_scm is computed on `feat.detach()`, so
                its gradient reaches f_x but never the backbone (preventing the
                backbone from degrading `feat` to be trivially modelled).

      Pass B -- classifier + causal regularizers.  f_x is frozen
                (`set_scm_requires_grad(False)`) so L_cf cannot collapse the
                mechanism into feat_cf == feat; it can only move the backbone
                and the disease pathway towards counterfactual invariance.

    `lambda_irm` and `lambda_cf` are linearly warmed up: IRM over the first
    `irm_warmup_epochs`, and L_cf over the first `cf_warmup_epochs` (the
    mechanism must be partly fitted before its counterfactuals are meaningful).

    train_loader / val_loader must yield
        (images, labels, ages, sexes, locs, age_groups).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if verbose:
        causal = [m for m in ("age", "sex", "loc") if m not in spurious_meta]
        print(f"  L_cf intervenes on spurious M = {tuple(spurious_meta)}  |  "
              f"causal M (left free) = {tuple(causal)}")

    best_acc = 0.0
    history = {k: [] for k in (
        "train_loss", "train_acc", "val_loss", "val_acc",
        "L_disease", "L_meta", "L_orth", "L_irm", "L_scm", "L_cf")}

    for epoch in range(1, epochs + 1):
        cur_lam_irm = lambda_irm * min(1.0, epoch / max(1, irm_warmup_epochs))
        cur_lam_cf  = lambda_cf  * min(1.0, epoch / max(1, cf_warmup_epochs))

        model.train()
        accum = {k: 0.0 for k in
                 ("loss", "dis", "meta", "orth", "irm", "scm", "cf")}
        correct, total = 0, 0

        pbar = tqdm(train_loader,
                    desc=f"[CausalSCM] Epoch {epoch}/{epochs}", leave=False)
        for images, labels, ages, sexes, locs, age_groups in pbar:
            images = images.to(device);   labels = labels.to(device)
            ages = ages.to(device);       sexes = sexes.to(device)
            locs = locs.to(device);       age_groups = age_groups.to(device)
            B = images.size(0)

            optimizer.zero_grad()

            # --- Backbone forward (once) ---
            feat = model.encode_feat(images)                  # (B, 1536)

            # ===== Pass A: maximum-likelihood fit of the mechanism f_x =====
            # feat.detach() -> L_scm trains f_x only, never the backbone.
            L_scm = model.structural_nll_loss(
                feat.detach(), labels, ages, sexes, locs)
            (lambda_scm * L_scm).backward()

            # ===== Pass B: classifier + causal regularizers =====
            disease_logits, sex_logits, loc_logits, age_logits, z_c, z_s = \
                model.decode(feat, ages, sexes, locs)

            # 1) disease classification
            L_dis = Fn.cross_entropy(disease_logits, labels)
            # 2) metadata heads (off Z_s)
            L_meta = (Fn.cross_entropy(sex_logits, sexes)
                      + Fn.cross_entropy(loc_logits, locs)
                      + Fn.cross_entropy(age_logits, age_groups))
            # 3) orthogonality  Z_c _|_ Z_s
            L_orth = (z_c * z_s).sum(dim=1).pow(2).mean()
            # 4) IRM across age environments
            L_irm = irm_penalty(disease_logits, labels, age_groups, num_envs)
            # 5) genuine counterfactual invariance on do(M_spurious) -- f_x FIXED
            model.set_scm_requires_grad(False)
            L_cf = counterfactual_invariance_loss(
                model, feat, labels, ages, sexes, locs,
                spurious=spurious_meta,
                disease_logits=disease_logits)
            model.set_scm_requires_grad(True)

            L_main = (L_dis
                      + lambda_meta * L_meta
                      + lambda_orth * L_orth
                      + cur_lam_irm * L_irm
                      + cur_lam_cf  * L_cf)
            L_main.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # --- bookkeeping ---
            loss_val = L_main.item() + lambda_scm * L_scm.item()
            accum["loss"] += loss_val * B
            accum["dis"]  += L_dis.item()  * B
            accum["meta"] += L_meta.item() * B
            accum["orth"] += L_orth.item() * B
            accum["irm"]  += (L_irm.item() if torch.is_tensor(L_irm)
                              else L_irm) * B
            accum["scm"]  += L_scm.item()  * B
            accum["cf"]   += L_cf.item()   * B
            total += B
            correct += disease_logits.argmax(1).eq(labels).sum().item()

            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(d=f"{L_dis.item():.3f}",
                                 scm=f"{L_scm.item():.1f}",
                                 cf=f"{L_cf.item():.4f}")

        train_acc = correct / max(1, total)
        for k in accum:
            accum[k] /= max(1, total)

        # --- validation (disease accuracy) ---
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, ages, sexes, locs, _ in val_loader:
                images = images.to(device); labels = labels.to(device)
                ages = ages.to(device); sexes = sexes.to(device)
                locs = locs.to(device)
                logits = model(images, ages, sexes, locs)[0]
                v_loss += Fn.cross_entropy(logits, labels).item() * images.size(0)
                v_total += labels.size(0)
                v_correct += logits.argmax(1).eq(labels).sum().item()
        val_loss = v_loss / max(1, v_total)
        val_acc  = v_correct / max(1, v_total)
        scheduler.step()

        history["train_loss"].append(accum["loss"])
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["L_disease"].append(accum["dis"])
        history["L_meta"].append(accum["meta"])
        history["L_orth"].append(accum["orth"])
        history["L_irm"].append(accum["irm"])
        history["L_scm"].append(accum["scm"])
        history["L_cf"].append(accum["cf"])

        if verbose:
            print(f"  Epoch {epoch}/{epochs}  "
                  f"Train={train_acc:.4f}  Val={val_acc:.4f}  "
                  f"D={accum['dis']:.3f}  M={accum['meta']:.3f}  "
                  f"Orth={accum['orth']:.4f}  IRM={accum['irm']:.2f}  "
                  f"SCM={accum['scm']:.1f}  CF={accum['cf']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"    -> Saved (Val Acc={best_acc:.4f})")

    return history, best_acc


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE}")
    NUM_CLASSES, NUM_SEX, NUM_LOC = 7, 3, 15
    net = CausalFactorizedNet(NUM_CLASSES, NUM_SEX, NUM_LOC).to(DEVICE)

    B = 4
    x   = torch.randn(B, 3, 224, 224, device=DEVICE)
    age = torch.rand(B, device=DEVICE)
    sex = torch.randint(0, NUM_SEX, (B,), device=DEVICE)
    loc = torch.randint(0, NUM_LOC, (B,), device=DEVICE)
    y   = torch.randint(0, NUM_CLASSES, (B,), device=DEVICE)

    net.eval()
    feat = net.encode_feat(x)
    print("feat:", tuple(feat.shape))

    # ---- abduction round-trip under the SAME m must be exact ----
    u_x   = net.abduct(feat, y, age, sex, loc)
    recon = net.structural_feat(u_x, y, age, sex, loc)
    print("abduction round-trip max|err|:",
          (recon - feat).abs().max().item(), "(~0 -- f_x invertible)")

    # ---- null intervention do(M=m) must be the identity ----
    feat_id = net.counterfactual_feat(feat, y, age, sex, loc, age, sex, loc)
    print("null-intervention   max|err|:",
          (feat_id - feat).abs().max().item(), "(~0)")

    # ---- do(M=m') with m' != m must move the feature ----
    feat_cf = net.counterfactual_feat(feat, y, age, sex, loc,
                                      age.flip(0), sex.flip(0), loc.flip(0))
    print("do(M=m') shift mean|Δfeat|:",
          (feat_cf - feat).abs().mean().item(), "(> 0)")

    # ---- L_scm must be trainable (SGD lowers it) ----
    net.train()
    L0 = net.structural_nll_loss(feat.detach(), y, age, sex, loc)
    opt = torch.optim.Adam(net.scm_parameters(), lr=1e-2)
    for _ in range(5):
        opt.zero_grad()
        net.structural_nll_loss(feat.detach(), y, age, sex, loc).backward()
        opt.step()
    L1 = net.structural_nll_loss(feat.detach(), y, age, sex, loc)
    print(f"L_scm: {L0.item():.1f} -> {L1.item():.1f}  "
          f"({'fits' if L1 < L0 else 'BUG: not decreasing'})")

    # ---- mini training loop on synthetic batches ----
    print("\n-- train_causal_factorized_scm smoke (2 epochs, synthetic) --")
    def fake_batch():
        return (torch.randn(B, 3, 224, 224),
                torch.randint(0, NUM_CLASSES, (B,)),
                torch.rand(B), torch.randint(0, NUM_SEX, (B,)),
                torch.randint(0, NUM_LOC, (B,)),
                torch.randint(0, 3, (B,)))
    loader = [fake_batch(), fake_batch()]
    hist, best = train_causal_factorized_scm(
        net, loader, loader, device=DEVICE, epochs=2, lr=1e-4,
        irm_warmup_epochs=1, cf_warmup_epochs=1,
        save_path="/tmp/_smoke_causal_scm.pth")
    assert all(torch.isfinite(torch.tensor(v)).all()
               for v in hist.values() if v), "non-finite history value"
    print(f"history keys: {list(hist.keys())}")
    print(f"best val acc: {best:.4f}")
    print("OK")
