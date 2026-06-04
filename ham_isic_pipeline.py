"""
Causal Representation Learning for Skin Lesion Classification (v2)
===================================================================
Revision of v1 (train_ISIC_2018_01.py) addressing reviewer concerns.
Changes from v1:

  C.2  Architecture fix — Z_c does NOT see metadata.
       v1 fed [feat ∥ meta_emb] to BOTH proj_c and proj_s, which by
       construction let Z_c trivially encode demographics (sex/loc probe
       on Z_c was 0.99). v2 routes only the image feature into proj_c,
       and the concatenated [feat ∥ meta_emb] only into proj_s.

  C.3  Honest renaming — "counterfactual" → "metadata-invariance".
       v1 called the regularizer L_cf and described it as a counterfactual
       consistency loss. By the proposed SCM (m → x), a true counterfactual
       on m must propagate through f_x; permuting m while holding feat
       fixed is closer to a marginal-independence constraint. v2 keeps
       the same functional form but renames it L_minv to remove the
       SCM-inconsistent claim. Section 2.6 of the manuscript should be
       updated in parallel.

  C.7  Two-version external test reporting.
       v1 mapped NaN sex/localization on the ISIC 2018 Task 3 test set
       to an "unknown" class, which inflates apparent epidemiological
       shift. v2 evaluates the external test set TWICE:
         (a) full set (same as v1) — for comparability.
         (b) known-only subset (sex ≠ unknown AND loc ≠ unknown) — to
             reveal whether the val→ext gap is genuine shift or a
             labeling artifact in the metadata column.

Architecture (v2):
  Image → EfficientNet-B3 → feat (1536)
  feat                   → proj_c → Z_c (512) → Disease classifier
  [feat ∥ meta_emb(128)] → proj_s → Z_s (512) → Metadata classifiers

Loss = L_disease + λ1·L_meta + λ2·L_orth + λ3·L_IRM + λ4·L_minv

Models trained:
  1) Image-Only      (baseline)
  2) Metadata-Only   (baseline)
  3) Causal Factorized (v2)

Saves to: history_ISIC2018_v2/
Cache (shared with v1 runs if available): cache_ISIC2018_shared/

Additional:
  STEP 11   : External test on ISIC2018 Task3 Test set (1511 images).
  STEP 11.5 : External test on the known-only subset (C.7).
  STEP 12   : Causal DAG — interpretability analysis.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as T
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from collections import Counter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Segmentation-guided Lesion Cropping
# ============================================================
def crop_lesion(image, mask, padding_ratio=0.1, target_size=(224, 224)):
    mask_np = np.array(mask)
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    if not rows.any() or not cols.any():
        w, h = image.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        image = image.crop((left, top, left + side, top + side))
        return image.resize(target_size, Image.BILINEAR)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = mask_np.shape
    pad_h = int((rmax - rmin) * padding_ratio)
    pad_w = int((cmax - cmin) * padding_ratio)

    rmin = max(0, rmin - pad_h)
    rmax = min(h - 1, rmax + pad_h)
    cmin = max(0, cmin - pad_w)
    cmax = min(w - 1, cmax + pad_w)

    cropped = image.crop((cmin, rmin, cmax + 1, rmax + 1))
    return cropped.resize(target_size, Image.BILINEAR)


# ============================================================
# 2. Center-crop (no segmentation) for test images
# ============================================================
def center_crop_resize(image, target_size=(224, 224)):
    """Center-square-crop then resize (used when no segmentation mask)."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize(target_size, Image.BILINEAR)

# ============================================================
# 3. Preprocessing: raw images → cached .pt
# ============================================================
def preprocess_and_cache(df, image_dir, seg_dir, dx_map, sex_map, loc_map,
                         target_size=(224, 224), cache_path="cached_data.pt"):
    if os.path.exists(cache_path):
        print(f"  Cache found: {cache_path} — loading...")
        return torch.load(cache_path)

    print(f"  Building cache: {cache_path}  ({len(df)} samples)...")
    to_tensor = T.ToTensor()

    images_list, labels_list = [], []
    ages_list, sexes_list, locs_list = [], [], []

    for idx in tqdm(range(len(df)), desc="  Preprocessing", leave=True):
        row = df.iloc[idx]
        image_id = row["image_id"]

        img = Image.open(os.path.join(image_dir, f"{image_id}.jpg")).convert("RGB")
        mask = Image.open(os.path.join(seg_dir, f"{image_id}_segmentation.png")).convert("L")
        img = crop_lesion(img, mask, target_size=target_size)

        images_list.append(to_tensor(img))
        labels_list.append(dx_map[row["dx"]])
        ages_list.append(row["age"] / 85.0)
        sexes_list.append(sex_map[row["sex"]])
        locs_list.append(loc_map[row["localization"]])

    data = {
        "images": torch.stack(images_list),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "ages":   torch.tensor(ages_list, dtype=torch.float32),
        "sexes":  torch.tensor(sexes_list, dtype=torch.long),
        "locs":   torch.tensor(locs_list, dtype=torch.long),
    }

    torch.save(data, cache_path)
    print(f"  Saved {cache_path}  (images shape: {data['images'].shape})")
    return data


def preprocess_test_cache(df, image_dir, dx_map, sex_map, loc_map,
                          target_size=(224, 224), cache_path="cached_test.pt"):
    """Preprocess test images WITHOUT segmentation masks (center-crop)."""
    if os.path.exists(cache_path):
        print(f"  Cache found: {cache_path} — loading...")
        return torch.load(cache_path)

    print(f"  Building test cache: {cache_path}  ({len(df)} samples)...")
    to_tensor = T.ToTensor()

    images_list, labels_list = [], []
    ages_list, sexes_list, locs_list = [], [], []

    skipped = 0
    for idx in tqdm(range(len(df)), desc="  Preprocessing test", leave=True):
        row = df.iloc[idx]
        image_id = row["image_id"]

        img_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        img = center_crop_resize(img, target_size=target_size)

        images_list.append(to_tensor(img))
        labels_list.append(dx_map[row["dx"]])
        ages_list.append(row["age"] / 85.0)
        sexes_list.append(sex_map.get(str(row["sex"]), sex_map["unknown"]))
        locs_list.append(loc_map.get(str(row["localization"]),
                                     loc_map.get("unknown", 0)))

    data = {
        "images": torch.stack(images_list),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "ages":   torch.tensor(ages_list, dtype=torch.float32),
        "sexes":  torch.tensor(sexes_list, dtype=torch.long),
        "locs":   torch.tensor(locs_list, dtype=torch.long),
    }

    if skipped:
        print(f"  Skipped {skipped} missing image(s)")

    torch.save(data, cache_path)
    print(f"  Saved {cache_path}  (images shape: {data['images'].shape})")
    return data


# ============================================================
# 4. Models
# ============================================================

class CausalFactorizedNet(nn.Module):
    """
    Causal Representation Learning with Factorized Features (v2).

    v2 change (C.2): Z_c is built ONLY from the image feature; metadata
    enters the network exclusively through proj_s. This prevents Z_c
    from trivially encoding demographics through a metadata input.

    Forward:
        Image → backbone → feat (1536)
        feat                  → proj_c → Z_c → disease_head
        [feat ∥ meta_emb]     → proj_s → Z_s → sex/loc/age heads

    Metadata-invariance regularizer (renamed from "counterfactual" in v1):
    permuting metadata within a batch should not change the disease
    prediction. This enforces marginal independence Z_c ⊥ M | X rather
    than a true counterfactual under do(M).
    """

    def __init__(self, num_classes=7, num_sex=3, num_loc=15,
                 num_age_groups=3, z_dim=512, meta_emb_dim=128):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()
        feat_dim = 1536

        # Metadata embedding
        self.age_fc = nn.Sequential(nn.Linear(1, 32), nn.ReLU())
        self.sex_emb = nn.Embedding(num_sex, 32)
        self.loc_emb = nn.Embedding(num_loc, 32)
        self.meta_fc = nn.Sequential(
            nn.Linear(96, meta_emb_dim), nn.ReLU(),
        )

        fused_dim = feat_dim + meta_emb_dim  # 1664 — used by proj_s only

        # C.2 — Causal projection: image feature ONLY, no metadata
        self.proj_c = nn.Sequential(
            nn.Linear(feat_dim, 768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, z_dim),
        )
        # Spurious projection: image + metadata (encodes shortcut info)
        self.proj_s = nn.Sequential(
            nn.Linear(fused_dim, 768), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(768, z_dim),
        )

        # Disease classifier (from Z_c only)
        self.disease_head = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )
        # Metadata classifiers (from Z_s)
        self.sex_head = nn.Linear(z_dim, num_sex)
        self.loc_head = nn.Linear(z_dim, num_loc)
        self.age_group_head = nn.Linear(z_dim, num_age_groups)

    def _embed_meta(self, age, sex, loc):
        age_e = self.age_fc(age.unsqueeze(-1))   # (B, 32)
        sex_e = self.sex_emb(sex)                # (B, 32)
        loc_e = self.loc_emb(loc)                # (B, 32)
        meta = torch.cat([age_e, sex_e, loc_e], dim=1)
        return self.meta_fc(meta)                # (B, 128)

    def decode(self, feat, age, sex, loc):
        """From cached backbone features + metadata → all outputs.

        v2: proj_c sees only `feat`; proj_s sees [feat ∥ meta_emb].
        """
        meta = self._embed_meta(age, sex, loc)
        fused_s = torch.cat([feat, meta], dim=1)   # (B, 1664) — for Z_s only

        z_c = self.proj_c(feat)                    # (B, 512) — image-only
        z_s = self.proj_s(fused_s)                 # (B, 512)

        disease_logits = self.disease_head(z_c)
        sex_logits = self.sex_head(z_s)
        loc_logits = self.loc_head(z_s)
        age_logits = self.age_group_head(z_s)

        return disease_logits, sex_logits, loc_logits, age_logits, z_c, z_s

    def forward(self, x, age, sex, loc):
        feat = self.backbone(x)
        return self.decode(feat, age, sex, loc)


class ImageOnlyNet(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
        self.backbone.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(1536, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


class MetaOnlyNet(nn.Module):
    def __init__(self, meta_dim, num_classes=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(meta_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Datasets
# ============================================================

class CachedImageDataset(Dataset):
    """Images from .pt cache. Returns (img, label, age, sex, loc, age_group)."""

    def __init__(self, data_dict, transform=None):
        self.images = data_dict["images"]
        self.labels = data_dict["labels"]
        self.ages   = data_dict["ages"]
        self.sexes  = data_dict["sexes"]
        self.locs   = data_dict["locs"]
        self.transform = transform

        # Compute age-group environments from normalized ages
        ages_real = self.ages * 85.0
        self.age_groups = torch.zeros(len(self.labels), dtype=torch.long)
        self.age_groups[ages_real >= 40] = 1
        self.age_groups[ages_real >= 60] = 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return (img, self.labels[idx], self.ages[idx],
                self.sexes[idx], self.locs[idx], self.age_groups[idx])


class CachedMetaDataset(Dataset):
    """Metadata-only dataset from pre-loaded tensors."""

    def __init__(self, data_dict, num_sex, num_loc):
        self.labels = data_dict["labels"]
        N = len(self.labels)
        self.meta = torch.zeros(N, 1 + num_sex + num_loc)
        self.meta[:, 0] = data_dict["ages"]
        for i in range(N):
            self.meta[i, 1 + data_dict["sexes"][i]] = 1.0
            self.meta[i, 1 + num_sex + data_dict["locs"][i]] = 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.meta[idx], self.labels[idx]


# ============================================================
# 5. IRM Penalty
# ============================================================

def irm_penalty(disease_logits, labels, env_ids, num_envs=3):
    """IRMv1 penalty: sum_e ||grad_w R_e(w)|_{w=1}||^2"""
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
# 6. Training Functions
# ============================================================

def train_causal_factorized(model, train_loader, val_loader, device, epochs, lr,
                             lambda_meta, lambda_orth, lambda_irm, lambda_minv,
                             irm_warmup_epochs, save_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "L_disease": [], "L_meta": [], "L_orth": [], "L_irm": [], "L_minv": [],
    }

    for epoch in range(1, epochs + 1):
        # IRM warm-up: linearly ramp lambda_irm over first N epochs
        if epoch <= irm_warmup_epochs:
            cur_lam_irm = lambda_irm * epoch / irm_warmup_epochs
        else:
            cur_lam_irm = lambda_irm

        model.train()
        accum = {k: 0.0 for k in ["loss", "dis", "meta", "orth", "irm", "minv"]}
        correct, total = 0, 0

        pbar = tqdm(train_loader,
                    desc=f"[CausalV3] Epoch {epoch}/{epochs}", leave=False)
        for images, labels, ages, sexes, locs, age_groups in pbar:
            images = images.to(device)
            labels = labels.to(device)
            ages = ages.to(device)
            sexes = sexes.to(device)
            locs = locs.to(device)
            age_groups = age_groups.to(device)
            B = images.size(0)

            optimizer.zero_grad()

            # --- Forward (backbone once, reuse feat) ---
            feat = model.backbone(images)                          # (B, 1536)
            disease_logits, sex_logits, loc_logits, age_logits, z_c, z_s = \
                model.decode(feat, ages, sexes, locs)

            # 1) Disease loss
            L_dis = Fn.cross_entropy(disease_logits, labels)

            # 2) Metadata loss
            L_meta = (Fn.cross_entropy(sex_logits, sexes)
                      + Fn.cross_entropy(loc_logits, locs)
                      + Fn.cross_entropy(age_logits, age_groups))

            # 3) Orthogonality: Z_c ⊥ Z_s
            L_orth = (z_c * z_s).sum(dim=1).pow(2).mean()

            # 4) IRM penalty across age environments
            L_irm = irm_penalty(disease_logits, labels, age_groups, num_envs=3)

            # 5) Metadata-invariance regularizer (renamed from L_cf in v1).
            #    Shuffle metadata within the batch while keeping `feat`
            #    fixed; predictions should remain stable. This is a
            #    marginal-independence constraint on Z_c, NOT a true
            #    counterfactual: a true do(M=m') would have to propagate
            #    through f_x : (M, ε_x) → X.
            perm = torch.randperm(B, device=device)
            disease_minv, _, _, _, _, _ = model.decode(
                feat, ages[perm], sexes[perm], locs[perm])
            L_minv = Fn.mse_loss(disease_logits.softmax(dim=1),
                                 disease_minv.softmax(dim=1))

            # Total loss
            loss = (L_dis
                    + lambda_meta * L_meta
                    + lambda_orth * L_orth
                    + cur_lam_irm * L_irm
                    + lambda_minv * L_minv)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Accumulate
            accum["loss"] += loss.item() * B
            accum["dis"]  += L_dis.item() * B
            accum["meta"] += L_meta.item() * B
            accum["orth"] += L_orth.item() * B
            accum["irm"]  += (L_irm.item() if torch.is_tensor(L_irm) else L_irm) * B
            accum["minv"] += L_minv.item() * B
            _, pred = disease_logits.max(1)
            total += B
            correct += pred.eq(labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             d=f"{L_dis.item():.3f}",
                             orth=f"{L_orth.item():.4f}",
                             minv=f"{L_minv.item():.4f}")

        train_acc = correct / total
        for k in accum:
            accum[k] /= total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, ages, sexes, locs, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                ages, sexes, locs = ages.to(device), sexes.to(device), locs.to(device)
                logits = model(images, ages, sexes, locs)[0]
                v_loss += Fn.cross_entropy(logits, labels).item() * images.size(0)
                v_total += labels.size(0)
                v_correct += logits.max(1)[1].eq(labels).sum().item()

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        scheduler.step()

        history["train_loss"].append(accum["loss"])
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["L_disease"].append(accum["dis"])
        history["L_meta"].append(accum["meta"])
        history["L_orth"].append(accum["orth"])
        history["L_irm"].append(accum["irm"])
        history["L_minv"].append(accum["minv"])

        print(f"  Epoch {epoch}/{epochs}  "
              f"Train={train_acc:.4f}  Val={val_acc:.4f}  "
              f"D={accum['dis']:.3f}  M={accum['meta']:.3f}  "
              f"Orth={accum['orth']:.4f}  IRM={accum['irm']:.2f}  "
              f"MInv={accum['minv']:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    -> Saved (Acc={best_acc:.4f})")

    return history, best_acc


def train_image_only(model, train_loader, val_loader, device, epochs, lr,
                     save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader,
                    desc=f"[ImgOnly] Epoch {epoch}/{epochs}", leave=False)
        for images, labels, *_ in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss, train_acc = run_loss / total, correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels, *_ in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                v_loss += loss.item() * images.size(0)
                v_total += labels.size(0)
                v_correct += logits.max(1)[1].eq(labels).sum().item()

        val_loss, val_acc = v_loss / v_total, v_correct / v_total
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch}/{epochs}  Train={train_acc:.4f}  Val={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    -> Saved (Acc={best_acc:.4f})")
    return history, best_acc


def train_meta_only(model, train_loader, val_loader, device, epochs, lr,
                    save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(train_loader,
                    desc=f"[MetaOnly] Epoch {epoch}/{epochs}", leave=False)
        for meta, labels in pbar:
            meta, labels = meta.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(meta)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * meta.size(0)
            _, pred = logits.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss, train_acc = run_loss / total, correct / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for meta, labels in val_loader:
                meta, labels = meta.to(device), labels.to(device)
                logits = model(meta)
                loss = criterion(logits, labels)
                v_loss += loss.item() * meta.size(0)
                v_total += labels.size(0)
                v_correct += logits.max(1)[1].eq(labels).sum().item()

        val_loss, val_acc = v_loss / v_total, v_correct / v_total
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch}/{epochs}  Train={train_acc:.4f}  Val={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    -> Saved (Acc={best_acc:.4f})")
    return history, best_acc


# ============================================================
# 7. Evaluation & Feature Extraction
# ============================================================

def evaluate_on_loader(model, loader, device, model_type="image_only"):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            if model_type == "meta_only":
                meta, labels = batch
                meta, labels = meta.to(device), labels.to(device)
                logits = model(meta)
            elif model_type == "image_only":
                images, labels = batch[0].to(device), batch[1].to(device)
                logits = model(images)
            elif model_type == "causal_factorized":
                images, labels, ages, sexes, locs, _ = batch
                images, labels = images.to(device), labels.to(device)
                ages = ages.to(device)
                sexes = sexes.to(device)
                locs = locs.to(device)
                logits = model(images, ages, sexes, locs)[0]

            probs = torch.softmax(logits, dim=1)
            all_preds.append(logits.max(1)[1].cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    return (torch.cat(all_preds).numpy(),
            torch.cat(all_probs).numpy(),
            torch.cat(all_labels).numpy())


def extract_z_c(model, loader, device):
    """Extract Z_c representations from CausalFactorizedNet."""
    model.eval()
    all_zc, all_labels, all_sexes, all_locs, all_ages = [], [], [], [], []
    with torch.no_grad():
        for images, labels, ages, sexes, locs, _ in loader:
            images = images.to(device)
            ages, sexes, locs = ages.to(device), sexes.to(device), locs.to(device)
            _, _, _, _, z_c, _ = model(images, ages, sexes, locs)
            all_zc.append(z_c.cpu())
            all_labels.append(labels)
            all_sexes.append(sexes.cpu())
            all_locs.append(locs.cpu())
            all_ages.append(ages.cpu())
    return {
        "z_c":    torch.cat(all_zc).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "sexes":  torch.cat(all_sexes).numpy(),
        "locs":   torch.cat(all_locs).numpy(),
        "ages":   torch.cat(all_ages).numpy(),
    }


def extract_backbone_features(model, loader, device):
    """Extract backbone features from ImageOnlyNet."""
    model.eval()
    all_feats, all_labels, all_sexes, all_locs, all_ages = [], [], [], [], []
    with torch.no_grad():
        for images, labels, ages, sexes, locs, _ in loader:
            images = images.to(device)
            feats = model.backbone(images)
            all_feats.append(feats.cpu())
            all_labels.append(labels)
            all_sexes.append(sexes)
            all_locs.append(locs)
            all_ages.append(ages)
    return {
        "features": torch.cat(all_feats).numpy(),
        "labels":   torch.cat(all_labels).numpy(),
        "sexes":    torch.cat(all_sexes).numpy(),
        "locs":     torch.cat(all_locs).numpy(),
        "ages":     torch.cat(all_ages).numpy(),
    }


def compute_metrics(preds, labels, num_classes):
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    rows = []
    for i in range(num_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        rows.append({
            "class_idx": i,
            "sensitivity": TP / (TP + FN + 1e-8),
            "specificity": TN / (TN + FP + 1e-8),
        })
    return acc, cm, pd.DataFrame(rows)


# ============================================================
# 8. Visualization Helpers
# ============================================================

def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_roc(probs, labels, num_classes, class_names, title, save_path):
    y_bin = label_binarize(labels, classes=list(range(num_classes)))
    fpr, tpr, roc_auc_dict = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc_dict[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", num_classes)
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"{class_names[i]} (AUC={roc_auc_dict[i]:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    return roc_auc_dict


def plot_learning_curves(history, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} — Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{title} — Accuracy"); axes[1].legend()
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_causal_loss_components(history, save_path):
    """Plot all 5 loss components of the Causal Factorized model."""
    components = ["L_disease", "L_meta", "L_orth", "L_irm", "L_minv"]
    titles = ["Disease CE", "Metadata CE", "Orthogonality", "IRM Penalty",
              "Metadata-Invariance"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for i, (comp, ttl) in enumerate(zip(components, titles)):
        ax = axes[i // 3][i % 3]
        ax.plot(history[comp], linewidth=2)
        ax.set_title(ttl); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    axes[1][2].axis("off")
    plt.suptitle("Causal Factorized — Loss Components", fontsize=14)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# ============================================================
# 9. Causal DAG — Visualization & Analysis
# ============================================================

def build_causal_dag():
    """
    Build the assumed causal DAG for skin lesion classification.

    Nodes:
        Y  = Disease (akiec, bcc, bkl, df, mel, nv, vasc)
        X  = Image (skin lesion photograph)
        A  = Age (patient age)
        S  = Sex (male, female, unknown)
        L  = Localization (body site)
        Z_c = Causal representation (learned)
        Z_s = Spurious representation (learned)

    Edges (medical prior knowledge):
        Y → X   : Disease determines lesion morphology (CAUSAL)
        A → Y   : Age is a risk factor for certain diseases
        S → Y   : Sex affects disease prevalence
        L → Y   : Some diseases are site-specific
        A → X   : Age affects skin appearance (confounding)
        S → X   : Sex affects skin characteristics (confounding)
        L → X   : Body site affects image (confounding)

    Model mapping:
        X  → Z_c : proj_c extracts causal features (disease morphology)
        X  → Z_s : proj_s extracts spurious features (metadata-correlated)
        Z_c → Y  : disease_head predicts disease from Z_c
        Z_s → S,L,A : metadata heads predict from Z_s
    """
    import networkx as nx

    G = nx.DiGraph()

    # Observed variables
    G.add_node("Y",  label="Disease (Y)", node_type="target",   layer=3)
    G.add_node("X",  label="Image (X)",   node_type="observed", layer=2)
    G.add_node("A",  label="Age (A)",     node_type="metadata", layer=0)
    G.add_node("S",  label="Sex (S)",     node_type="metadata", layer=0)
    G.add_node("L",  label="Loc (L)",     node_type="metadata", layer=0)

    # Learned representations
    G.add_node("Zc", label="Z_c\n(causal)", node_type="latent", layer=1)
    G.add_node("Zs", label="Z_s\n(spurious)", node_type="latent", layer=1)

    # Causal edges (medical knowledge)
    G.add_edge("Y", "X",  edge_type="causal",
               desc="Disease determines lesion morphology")
    G.add_edge("A", "Y",  edge_type="confounder",
               desc="Age is a risk factor")
    G.add_edge("S", "Y",  edge_type="confounder",
               desc="Sex affects prevalence")
    G.add_edge("L", "Y",  edge_type="confounder",
               desc="Site-specific diseases")
    G.add_edge("A", "X",  edge_type="confounding",
               desc="Age affects skin appearance")
    G.add_edge("S", "X",  edge_type="confounding",
               desc="Sex affects skin features")
    G.add_edge("L", "X",  edge_type="confounding",
               desc="Body site affects image")

    # Model edges (learned)
    G.add_edge("X",  "Zc", edge_type="model",
               desc="proj_c extracts causal features")
    G.add_edge("X",  "Zs", edge_type="model",
               desc="proj_s extracts spurious features")
    G.add_edge("Zc", "Y",  edge_type="model",
               desc="disease_head: Z_c → prediction")
    G.add_edge("Zs", "S",  edge_type="model",
               desc="sex_head: Z_s → sex prediction")
    G.add_edge("Zs", "L",  edge_type="model",
               desc="loc_head: Z_s → loc prediction")
    G.add_edge("Zs", "A",  edge_type="model",
               desc="age_head: Z_s → age prediction")

    return G


def plot_causal_dag(G, save_path, edge_strengths=None):
    """Draw the causal DAG with color-coded edges and annotations."""
    import networkx as nx

    fig, ax = plt.subplots(figsize=(16, 12))

    # Manual positions for clarity
    pos = {
        "A":  (-2.5, 3.0),
        "S":  (0.0,  3.0),
        "L":  (2.5,  3.0),
        "X":  (0.0,  1.0),
        "Y":  (0.0, -1.5),
        "Zc": (-2.0, -0.5),
        "Zs": (2.0,  -0.5),
    }

    # Node colors by type
    node_colors = {
        "target":   "#E74C3C",  # red
        "observed": "#3498DB",  # blue
        "metadata": "#2ECC71",  # green
        "latent":   "#F39C12",  # orange
    }
    colors = [node_colors[G.nodes[n]["node_type"]] for n in G.nodes]
    labels = {n: G.nodes[n]["label"] for n in G.nodes}

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                           node_size=3500, alpha=0.9, edgecolors="black",
                           linewidths=2)
    nx.draw_networkx_labels(G, pos, labels, ax=ax,
                            font_size=11, font_weight="bold")

    # Edge styles by type
    edge_styles = {
        "causal":     {"color": "#E74C3C", "style": "solid",  "width": 3.0},
        "confounder":  {"color": "#2ECC71", "style": "solid",  "width": 2.0},
        "confounding": {"color": "#2ECC71", "style": "dashed", "width": 2.0},
        "model":       {"color": "#3498DB", "style": "solid",  "width": 2.0},
    }

    for u, v, data in G.edges(data=True):
        etype = data["edge_type"]
        style = edge_styles[etype]
        label = ""
        if edge_strengths and (u, v) in edge_strengths:
            label = f"{edge_strengths[(u, v)]:.3f}"

        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=style["color"], style=style["style"],
            width=style["width"], alpha=0.8,
            arrows=True, arrowsize=25, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            min_source_margin=30, min_target_margin=30,
        )
        if label:
            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2
            ax.text(mid_x + 0.15, mid_y + 0.15, label,
                    fontsize=9, color=style["color"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=style["color"], alpha=0.8))

    # Legend
    from matplotlib.patches import Patch, FancyArrowPatch
    legend_elements = [
        Patch(facecolor="#E74C3C", label="Target (Y)"),
        Patch(facecolor="#3498DB", label="Observed (X)"),
        Patch(facecolor="#2ECC71", label="Metadata (A, S, L)"),
        Patch(facecolor="#F39C12", label="Latent (Z_c, Z_s)"),
        plt.Line2D([0], [0], color="#E74C3C", lw=3, label="Causal: Y→X"),
        plt.Line2D([0], [0], color="#2ECC71", lw=2, label="Confounder: Meta→Y"),
        plt.Line2D([0], [0], color="#2ECC71", lw=2, ls="--",
                   label="Confounding: Meta→X"),
        plt.Line2D([0], [0], color="#3498DB", lw=2, label="Model: learned"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
              framealpha=0.9)

    ax.set_title("Causal DAG — Skin Lesion Classification\n"
                 "(HAM10000 / ISIC2018)", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def compute_edge_strengths(df, dx_map, sex_map, loc_map):
    """
    Compute empirical association strengths for DAG edges using
    Cramér's V (categorical×categorical) and correlation ratio (continuous×categorical).

    Returns dict: {(source, target): strength_value}
    """
    from scipy.stats import chi2_contingency

    def cramers_v(x, y):
        """Cramér's V for two categorical variables."""
        ct = pd.crosstab(x, y)
        chi2 = chi2_contingency(ct)[0]
        n = ct.sum().sum()
        k = min(ct.shape) - 1
        if k == 0 or n == 0:
            return 0.0
        return np.sqrt(chi2 / (n * k))

    def correlation_ratio(categories, values):
        """Eta-squared: how much variance in values is explained by categories."""
        categories = np.asarray(categories)
        values = np.asarray(values, dtype=float)
        grand_mean = values.mean()
        ss_between = 0.0
        ss_total = ((values - grand_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        for cat in np.unique(categories):
            mask = categories == cat
            group_mean = values[mask].mean()
            ss_between += mask.sum() * (group_mean - grand_mean) ** 2
        return np.sqrt(ss_between / ss_total)

    strengths = {}

    # A → Y: correlation ratio (age → disease)
    strengths[("A", "Y")] = correlation_ratio(df["dx"], df["age"])

    # S → Y: Cramér's V (sex × disease)
    strengths[("S", "Y")] = cramers_v(df["sex"], df["dx"])

    # L → Y: Cramér's V (localization × disease)
    strengths[("L", "Y")] = cramers_v(df["localization"], df["dx"])

    # A → X: we can't directly compute image correlation, use age→localization
    # as a proxy for how age correlates with visible features
    strengths[("A", "X")] = correlation_ratio(df["localization"], df["age"])

    # S → X: proxy via sex→localization
    strengths[("S", "X")] = cramers_v(df["sex"], df["localization"])

    # L → X: localization directly determines image appearance (strong by definition)
    strengths[("L", "X")] = 1.0  # by definition, body site determines image context

    # Y → X: disease determines morphology (strong by definition)
    strengths[("Y", "X")] = 1.0

    return strengths


def compute_conditional_probs(df, dx_map, sex_map, loc_map, save_dir):
    """
    Compute and save conditional probability tables for key DAG edges:
      P(Y|A), P(Y|S), P(Y|L) — how metadata predicts disease
    """
    results = {}

    # --- P(Y | Sex) ---
    ct_sex = pd.crosstab(df["sex"], df["dx"], normalize="index")
    ct_sex.to_csv(os.path.join(save_dir, "dag_P_Y_given_Sex.csv"))
    results["P(Y|Sex)"] = ct_sex

    # --- P(Y | Age group) ---
    df_temp = df.copy()
    df_temp["age_group"] = pd.cut(df_temp["age"], bins=[0, 40, 60, 100],
                                  labels=["<40", "40-60", ">=60"])
    ct_age = pd.crosstab(df_temp["age_group"], df_temp["dx"], normalize="index")
    ct_age.to_csv(os.path.join(save_dir, "dag_P_Y_given_AgeGroup.csv"))
    results["P(Y|AgeGroup)"] = ct_age

    # --- P(Y | Localization) — top 10 sites ---
    top_locs = df["localization"].value_counts().head(10).index
    df_top = df[df["localization"].isin(top_locs)]
    ct_loc = pd.crosstab(df_top["localization"], df_top["dx"], normalize="index")
    ct_loc.to_csv(os.path.join(save_dir, "dag_P_Y_given_Loc.csv"))
    results["P(Y|Loc)"] = ct_loc

    # --- P(Loc | Sex) — confounding path via shared cause ---
    ct_loc_sex = pd.crosstab(df["sex"], df["localization"], normalize="index")
    ct_loc_sex.to_csv(os.path.join(save_dir, "dag_P_Loc_given_Sex.csv"))
    results["P(Loc|Sex)"] = ct_loc_sex

    return results


def plot_conditional_heatmaps(cond_probs, save_dir):
    """Plot heatmaps for conditional probability tables."""
    for name, table in cond_probs.items():
        fig, ax = plt.subplots(figsize=(12, max(4, len(table) * 0.6)))
        sns.heatmap(table, annot=True, fmt=".3f", cmap="YlOrRd",
                    ax=ax, linewidths=0.5)
        ax.set_title(f"Conditional Probability: {name}", fontsize=13)
        ax.set_ylabel(name.split("|")[1].rstrip(")"))
        ax.set_xlabel("Disease (dx)")
        plt.tight_layout()
        fname = name.replace("(", "").replace(")", "").replace("|", "_given_")
        plt.savefig(os.path.join(save_dir, f"dag_{fname}.png"), dpi=150)
        plt.close()


def plot_model_dag_alignment(probe_results, shuffle_results, orth_loss,
                             minv_loss, save_path):
    """
    Visualize how each loss component maps to blocking a DAG path.

    Layout: Table showing {DAG path → Loss component → Metric → Status}
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    # Table data
    col_labels = ["Causal Path", "Blocked By", "Metric", "Value",
                  "Target", "Status"]
    table_data = []

    # 1. Z_c ⊥ Z_s (orthogonality)
    status_orth = "PASS" if orth_loss < 0.01 else "PARTIAL" if orth_loss < 0.1 else "FAIL"
    table_data.append([
        "Z_c ↔ Z_s", "L_orth", "Orthogonality Loss",
        f"{orth_loss:.4f}", "< 0.01", status_orth
    ])

    # 2. Z_c ⊥ Sex (linear probe)
    if len(probe_results) >= 2:
        zc_sex = probe_results[1].get("Sex Probe Acc", 0)
        rand_sex = probe_results[2].get("Sex Probe Acc", 0) if len(probe_results) > 2 else 0.5
        status_sex = "PASS" if abs(zc_sex - rand_sex) < 0.05 else "PARTIAL" if abs(zc_sex - rand_sex) < 0.1 else "FAIL"
        table_data.append([
            "Sex → Z_c", "L_orth + L_minv", "Sex Probe Acc (Z_c)",
            f"{zc_sex:.4f}", f"~{rand_sex:.3f} (random)", status_sex
        ])

        zc_loc = probe_results[1].get("Loc Probe Acc", 0)
        rand_loc = probe_results[2].get("Loc Probe Acc", 0) if len(probe_results) > 2 else 0.1
        status_loc = "PASS" if abs(zc_loc - rand_loc) < 0.05 else "PARTIAL" if abs(zc_loc - rand_loc) < 0.1 else "FAIL"
        table_data.append([
            "Loc → Z_c", "L_orth + L_minv", "Loc Probe Acc (Z_c)",
            f"{zc_loc:.4f}", f"~{rand_loc:.3f} (random)", status_loc
        ])

    # 3. Metadata-invariance (renamed from "counterfactual" in v1)
    status_minv = "PASS" if minv_loss < 0.005 else "PARTIAL" if minv_loss < 0.02 else "FAIL"
    table_data.append([
        "Meta → P(Y|X)", "L_minv", "MInv Loss (shuffle meta)",
        f"{minv_loss:.4f}", "< 0.005", status_minv
    ])

    # 4. Shuffle test drop
    if len(shuffle_results) >= 2:
        zc_drop_sex = shuffle_results[1].get("Sex Drop", 0)
        zc_drop_loc = shuffle_results[1].get("Loc Drop", 0)
        img_drop_sex = shuffle_results[0].get("Sex Drop", 0)
        status_shuf = "PASS" if zc_drop_sex < img_drop_sex * 0.5 else "PARTIAL"
        table_data.append([
            "Meta encoding", "All causal losses", "Shuffle Sex Drop (Z_c)",
            f"{zc_drop_sex:.4f}", f"< {img_drop_sex:.3f} (ImgOnly)", status_shuf
        ])

    # Draw table
    cell_colors = []
    for row in table_data:
        row_colors = ["#f8f9fa"] * 5
        status = row[-1]
        if status == "PASS":
            row_colors.append("#d4edda")
        elif status == "PARTIAL":
            row_colors.append("#fff3cd")
        else:
            row_colors.append("#f8d7da")
        cell_colors.append(row_colors)

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     cellColours=cell_colors,
                     colColours=["#343a40"] * len(col_labels),
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)

    # Color header text white
    for j in range(len(col_labels)):
        table[0, j].get_text().set_color("white")
        table[0, j].get_text().set_fontweight("bold")

    ax.set_title("Model-DAG Alignment: Causal Path Blocking Status\n"
                 "(How well each loss component blocks spurious paths)",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dag_intervention_summary(save_path):
    """
    Draw an annotated DAG showing which model components
    intervene on which causal paths.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # Node positions
    pos = {
        "Age":   (-3, 4),  "Sex":  (0, 4),   "Loc":  (3, 4),
        "Image": (0, 2),
        "Z_c":   (-2.5, 0), "Z_s":  (2.5, 0),
        "Disease": (0, -2),
    }

    # Draw nodes
    node_cfg = {
        "Age":     ("#2ECC71", "Age\n(A)"),
        "Sex":     ("#2ECC71", "Sex\n(S)"),
        "Loc":     ("#2ECC71", "Loc\n(L)"),
        "Image":   ("#3498DB", "Image\n(X)"),
        "Z_c":     ("#F39C12", "Z_c\n(causal)"),
        "Z_s":     ("#F39C12", "Z_s\n(spurious)"),
        "Disease": ("#E74C3C", "Disease\n(Y)"),
    }

    for node, (color, label) in node_cfg.items():
        x, y = pos[node]
        circle = plt.Circle((x, y), 0.6, color=color, alpha=0.85,
                             ec="black", linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=10, fontweight="bold")

    # Arrows
    def draw_arrow(src, dst, color, style="-", lw=2, label="", offset=0):
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        ux, uy = dx / length, dy / length
        # Start/end outside circles
        sx, sy = x1 + ux * 0.65, y1 + uy * 0.65
        ex, ey = x2 - ux * 0.65, y2 - uy * 0.65
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=lw, ls=style, mutation_scale=20))
        if label:
            mx = (sx + ex) / 2 + offset
            my = (sy + ey) / 2
            ax.text(mx, my, label, fontsize=8, color=color,
                    fontweight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec=color, alpha=0.85))

    # Causal: Disease → Image
    draw_arrow("Disease", "Image", "#E74C3C", lw=3, label="Y→X\n(causal)")

    # Confounders: Meta → Disease
    draw_arrow("Age", "Disease", "#2ECC71", label="A→Y", offset=-0.4)
    draw_arrow("Sex", "Disease", "#2ECC71", label="S→Y")
    draw_arrow("Loc", "Disease", "#2ECC71", label="L→Y", offset=0.4)

    # Confounding: Meta → Image
    draw_arrow("Age", "Image", "#2ECC71", style="--", label="A→X", offset=-0.3)
    draw_arrow("Sex", "Image", "#2ECC71", style="--")
    draw_arrow("Loc", "Image", "#2ECC71", style="--", label="L→X", offset=0.3)

    # Model: Image → Z_c, Z_s
    draw_arrow("Image", "Z_c", "#3498DB", lw=2.5, label="proj_c")
    draw_arrow("Image", "Z_s", "#3498DB", lw=2.5, label="proj_s")

    # Model: Z_c → Disease
    draw_arrow("Z_c", "Disease", "#E74C3C", lw=2.5, label="disease_head")

    # Model: Z_s → metadata
    draw_arrow("Z_s", "Sex", "#9B59B6", lw=2, label="sex_head")
    draw_arrow("Z_s", "Loc", "#9B59B6", lw=2, label="loc_head")
    draw_arrow("Z_s", "Age", "#9B59B6", lw=2, label="age_head")

    # Annotations for interventions (blocked paths)
    interventions = [
        (-0.3, -0.8, "L_orth: Z_c ⊥ Z_s\n(orthogonality)", "#E67E22"),
        (0.0,  -3.2, "L_minv: P(Y|X,M) ≈ P(Y|X,M')\n(metadata invariance)",
         "#8E44AD"),
        (-4.5, -1.5, "L_IRM: invariant across\nage environments", "#C0392B"),
        (4.5,  1.5,  "L_meta: Z_s encodes\nmetadata only", "#27AE60"),
    ]
    for x, y, text, color in interventions:
        ax.text(x, y, text, fontsize=9, color="white", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.4", fc=color, alpha=0.9))

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-4, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Causal DAG with Model Interventions\n"
                 "Skin Lesion Classification — Causal Factorized Architecture",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# 10. Main Pipeline
# ============================================================
if __name__ == "__main__":

    # ---- Paths ----
    IMAGE_DIR = "data/HAM10000_images_combined_600x450"
    SEG_DIR   = "data/HAM10000_segmentations_lesion_tschandl"
    META_CSV  = "data/HAM10000_metadata.csv"
    # External test set (ISIC2018 Task3 with GroundTruth)
    TEST_IMAGE_DIR = "data/ISIC2018_Task3_Test_Images"
    TEST_GT_CSV    = "data/ISIC2018_Task3_Test_GroundTruth.csv"
    SAVE_DIR  = "history_ISIC2018_v2"
    # Shared cache (top-level) so v1 and v2 reuse the same .pt files.
    CACHE_DIR = "cache_ISIC2018_shared"
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Backward-compat: if v1 produced caches inside history_ISIC2018_v1/cache,
    # symlink them into the shared cache so we don't re-preprocess.
    _v1_cache = os.path.join("history_ISIC2018_v1", "cache")
    if os.path.isdir(_v1_cache):
        for _fname in ("train_data.pt", "val_data.pt", "test_external_data.pt"):
            _src = os.path.abspath(os.path.join(_v1_cache, _fname))
            _dst = os.path.join(CACHE_DIR, _fname)
            if os.path.exists(_src) and not os.path.exists(_dst):
                try:
                    os.symlink(_src, _dst)
                    print(f"  [cache] symlinked {_fname} from v1 cache")
                except OSError:
                    import shutil
                    shutil.copy(_src, _dst)
                    print(f"  [cache] copied {_fname} from v1 cache")

    # ---- Hyperparameters ----
    BATCH_SIZE   = 32
    EPOCHS       = 50
    LR           = 1e-4
    LAMBDA_META  = 0.1       # λ1: metadata prediction weight
    LAMBDA_ORTH  = 0.01      # λ2: orthogonality Z_c ⊥ Z_s
    LAMBDA_IRM   = 1.0       # λ3: IRM penalty (after warm-up)
    LAMBDA_MINV  = 0.1       # λ4: metadata-invariance regularizer (was λ_cf in v1)
    IRM_WARMUP   = 5         # linearly ramp IRM over first N epochs
    TARGET_SIZE  = (224, 224)

    # ---- Load & split metadata ----
    df = pd.read_csv(META_CSV)
    df["age"] = df["age"].fillna(df["age"].mean())

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=42
    )

    dx_map  = {dx: i for i, dx in enumerate(sorted(df["dx"].unique()))}
    sex_map = {"male": 0, "female": 1, "unknown": 2}
    loc_map = {loc: i for i, loc in enumerate(sorted(df["localization"].unique()))}

    num_classes = len(dx_map)
    num_sex     = len(sex_map)
    num_loc     = len(loc_map)
    meta_dim    = 1 + num_sex + num_loc
    num_age_groups = 3

    class_names_inv  = {v: k for k, v in dx_map.items()}
    class_names_list = [class_names_inv[i] for i in range(num_classes)]

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
    print(f"Disease classes ({num_classes}): {dx_map}")
    print(f"Meta dim: {meta_dim}  |  Age groups: {num_age_groups}")

    # =========================================================
    # STEP 1: Preprocess images → .pt cache
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 1: PREPROCESSING — Images → .pt cache")
    print("=" * 60)

    train_cache = os.path.join(CACHE_DIR, "train_data.pt")
    val_cache   = os.path.join(CACHE_DIR, "val_data.pt")

    print("\n[Train set]")
    train_data = preprocess_and_cache(
        train_df, IMAGE_DIR, SEG_DIR, dx_map, sex_map, loc_map,
        target_size=TARGET_SIZE, cache_path=train_cache,
    )
    print("[Val set]")
    val_data = preprocess_and_cache(
        val_df, IMAGE_DIR, SEG_DIR, dx_map, sex_map, loc_map,
        target_size=TARGET_SIZE, cache_path=val_cache,
    )

    # =========================================================
    # STEP 2: Datasets & Loaders
    # =========================================================
    train_aug = T.Compose([
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_aug = T.Compose([
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_img_ds = CachedImageDataset(train_data, transform=train_aug)
    val_img_ds   = CachedImageDataset(val_data,   transform=val_aug)
    train_img_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=0, pin_memory=True)
    val_img_loader   = DataLoader(val_img_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, pin_memory=True)

    train_meta_ds = CachedMetaDataset(train_data, num_sex, num_loc)
    val_meta_ds   = CachedMetaDataset(val_data,   num_sex, num_loc)
    train_meta_loader = DataLoader(train_meta_ds, batch_size=BATCH_SIZE,
                                   shuffle=True, num_workers=0, pin_memory=True)
    val_meta_loader   = DataLoader(val_meta_ds, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=0, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Print age-group distribution
    ag_train = train_img_ds.age_groups
    ag_val   = val_img_ds.age_groups
    for split, ag in [("Train", ag_train), ("Val", ag_val)]:
        counts = [(ag == e).sum().item() for e in range(3)]
        print(f"  {split} age envs: <40={counts[0]}  40-60={counts[1]}  >=60={counts[2]}")

    # =========================================================
    # STEP 3: Train Image-Only (baseline)
    # =========================================================
    print("\n" + "=" * 60)
    print("MODEL 1: IMAGE-ONLY (EfficientNet-B3)")
    print("=" * 60)
    model_img = ImageOnlyNet(num_classes=num_classes).to(device)
    hist_img, best_img = train_image_only(
        model_img, train_img_loader, val_img_loader, device,
        epochs=EPOCHS, lr=LR,
        save_path=os.path.join(SAVE_DIR, "best_image_only.pth"),
    )
    pd.DataFrame(hist_img).to_csv(
        os.path.join(SAVE_DIR, "history_image_only.csv"), index=False)

    # =========================================================
    # STEP 4: Train Metadata-Only (baseline)
    # =========================================================
    print("\n" + "=" * 60)
    print("MODEL 2: METADATA-ONLY (MLP)")
    print("=" * 60)
    model_meta = MetaOnlyNet(meta_dim=meta_dim, num_classes=num_classes).to(device)
    hist_meta, best_meta = train_meta_only(
        model_meta, train_meta_loader, val_meta_loader, device,
        epochs=EPOCHS, lr=1e-3,
        save_path=os.path.join(SAVE_DIR, "best_meta_only.pth"),
    )
    pd.DataFrame(hist_meta).to_csv(
        os.path.join(SAVE_DIR, "history_meta_only.csv"), index=False)

    # =========================================================
    # STEP 5: Train Causal Factorized (main model)
    # =========================================================
    print("\n" + "=" * 60)
    print("MODEL 3: CAUSAL FACTORIZED (IRM + Orth + CF)")
    print("=" * 60)
    print(f"  Lambdas: meta={LAMBDA_META}  orth={LAMBDA_ORTH}  "
          f"irm={LAMBDA_IRM}  minv={LAMBDA_MINV}")
    print(f"  IRM warm-up: {IRM_WARMUP} epochs")

    model_causal = CausalFactorizedNet(
        num_classes=num_classes, num_sex=num_sex, num_loc=num_loc,
        num_age_groups=num_age_groups,
    ).to(device)
    hist_causal, best_causal = train_causal_factorized(
        model_causal, train_img_loader, val_img_loader, device,
        epochs=EPOCHS, lr=LR,
        lambda_meta=LAMBDA_META, lambda_orth=LAMBDA_ORTH,
        lambda_irm=LAMBDA_IRM, lambda_minv=LAMBDA_MINV,
        irm_warmup_epochs=IRM_WARMUP,
        save_path=os.path.join(SAVE_DIR, "best_causal_factorized.pth"),
    )
    pd.DataFrame(hist_causal).to_csv(
        os.path.join(SAVE_DIR, "history_causal_factorized.csv"), index=False)

    # =========================================================
    # STEP 6: Evaluate & Compare all 3 models
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 6: EVALUATION & COMPARISON")
    print("=" * 60)

    results = {}

    # Image-Only
    model_img.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_image_only.pth")))
    preds_i, probs_i, labels_i = evaluate_on_loader(
        model_img, val_img_loader, device, "image_only")
    acc_i, cm_i, ss_i = compute_metrics(preds_i, labels_i, num_classes)
    ss_i["class_name"] = [class_names_list[i] for i in ss_i["class_idx"]]
    results["Image-Only"] = {"acc": acc_i, "cm": cm_i, "ss": ss_i,
                             "preds": preds_i, "probs": probs_i, "labels": labels_i}

    # Meta-Only
    model_meta.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_meta_only.pth")))
    preds_m, probs_m, labels_m = evaluate_on_loader(
        model_meta, val_meta_loader, device, "meta_only")
    acc_m, cm_m, ss_m = compute_metrics(preds_m, labels_m, num_classes)
    ss_m["class_name"] = [class_names_list[i] for i in ss_m["class_idx"]]
    results["Meta-Only"] = {"acc": acc_m, "cm": cm_m, "ss": ss_m,
                            "preds": preds_m, "probs": probs_m, "labels": labels_m}

    # Causal Factorized
    model_causal.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, "best_causal_factorized.pth")))
    preds_c, probs_c, labels_c = evaluate_on_loader(
        model_causal, val_img_loader, device, "causal_factorized")
    acc_c, cm_c, ss_c = compute_metrics(preds_c, labels_c, num_classes)
    ss_c["class_name"] = [class_names_list[i] for i in ss_c["class_idx"]]
    results["Causal-Factorized"] = {"acc": acc_c, "cm": cm_c, "ss": ss_c,
                                    "preds": preds_c, "probs": probs_c,
                                    "labels": labels_c}

    # =========================================================
    # STEP 7: Plots
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 7: PLOTS")
    print("=" * 60)

    for name, key, hist in [("Image-Only", "image_only", hist_img),
                            ("Meta-Only", "meta_only", hist_meta),
                            ("Causal-Factorized", "causal_factorized", hist_causal)]:
        r = results[name]
        plot_confusion_matrix(r["cm"], class_names_list,
                              f"Confusion Matrix — {name}",
                              os.path.join(SAVE_DIR, f"cm_{key}.png"))
        plot_roc(r["probs"], r["labels"], num_classes, class_names_list,
                 f"ROC — {name}",
                 os.path.join(SAVE_DIR, f"roc_{key}.png"))
        plot_learning_curves(hist, name,
                             os.path.join(SAVE_DIR, f"lc_{key}.png"))
        r["ss"].to_csv(os.path.join(SAVE_DIR, f"sens_spec_{key}.csv"),
                       index=False)

    # Causal loss components
    plot_causal_loss_components(hist_causal,
                               os.path.join(SAVE_DIR, "causal_loss_components.png"))

    # Accuracy bar chart
    model_names = list(results.keys())
    accs = [results[n]["acc"] for n in model_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accs, color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, a in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{a:.4f}", ha="center", fontweight="bold")
    plt.ylabel("Validation Accuracy")
    plt.title("Model Comparison — Accuracy")
    plt.ylim(0, 1.05); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "comparison_accuracy.png"), dpi=150)
    plt.close()

    # Val accuracy curves overlay
    plt.figure(figsize=(10, 6))
    plt.plot(hist_img["val_acc"], label="Image-Only", linewidth=2)
    plt.plot(hist_meta["val_acc"], label="Meta-Only", linewidth=2)
    plt.plot(hist_causal["val_acc"], label="Causal-Factorized", linewidth=2)
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "comparison_val_acc_curves.png"), dpi=150)
    plt.close()

    # Comparison tables
    comparison_rows = []
    for name in model_names:
        r = results[name]
        for _, row in r["ss"].iterrows():
            comparison_rows.append({
                "Model": name, "Class": row["class_name"],
                "Sensitivity": row["sensitivity"],
                "Specificity": row["specificity"],
            })
    pd.DataFrame(comparison_rows).to_csv(
        os.path.join(SAVE_DIR, "comparison_sens_spec.csv"), index=False)

    summary_rows = []
    for name in model_names:
        r = results[name]
        summary_rows.append({
            "Model": name, "Val Accuracy": r["acc"],
            "Mean Sensitivity": r["ss"]["sensitivity"].mean(),
            "Mean Specificity": r["ss"]["specificity"].mean(),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(SAVE_DIR, "comparison_summary.csv"),
                      index=False)

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    # =========================================================
    # STEP 8: Linear Probe on Z_c
    #   Goal: Z_c should NOT predict metadata (sex/location)
    #   Compare: Image-Only backbone features vs Causal Z_c
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 8: LINEAR PROBE — Metadata predictability from features")
    print("=" * 60)

    print("\nExtracting features...")
    feat_data = extract_backbone_features(model_img, val_img_loader, device)
    zc_data   = extract_z_c(model_causal, val_img_loader, device)

    feats_img_val = feat_data["features"]     # (N, 1536)
    z_c_val       = zc_data["z_c"]            # (N, 512)
    sexes_val     = feat_data["sexes"]
    locs_val      = feat_data["locs"]

    print(f"  Image-Only features: {feats_img_val.shape}")
    print(f"  Causal Z_c:          {z_c_val.shape}")

    # --- 8A: Probe accuracy ---
    print("\n--- Linear Probe: Sex & Localization from features ---")
    probe_results = []
    for feat_name, feats in [("Image-Only (backbone)", feats_img_val),
                             ("Causal (Z_c)", z_c_val)]:
        clf_sex = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        clf_sex.fit(feats, sexes_val)
        sex_acc = accuracy_score(sexes_val, clf_sex.predict(feats))

        clf_loc = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        clf_loc.fit(feats, locs_val)
        loc_acc = accuracy_score(locs_val, clf_loc.predict(feats))

        probe_results.append({
            "Model": feat_name,
            "Sex Probe Acc": sex_acc,
            "Loc Probe Acc": loc_acc,
        })
        print(f"  {feat_name}:  Sex={sex_acc:.4f}  Loc={loc_acc:.4f}")

    # Random baseline
    sex_majority = max(Counter(sexes_val).values()) / len(sexes_val)
    loc_majority = max(Counter(locs_val).values()) / len(locs_val)
    probe_results.append({
        "Model": "Random (majority)",
        "Sex Probe Acc": sex_majority,
        "Loc Probe Acc": loc_majority,
    })
    print(f"  Random baseline:  Sex={sex_majority:.4f}  Loc={loc_majority:.4f}")

    probe_df = pd.DataFrame(probe_results)
    probe_df.to_csv(os.path.join(SAVE_DIR, "metadata_probe.csv"), index=False)

    # Probe bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    models_p = probe_df["Model"].tolist()
    colors_p = ["#4C72B0", "#C44E52", "#AAAAAA"]
    for ax, col, title in [(axes[0], "Sex Probe Acc", "Sex Predictability"),
                           (axes[1], "Loc Probe Acc", "Loc Predictability")]:
        ax.bar(models_p, probe_df[col], color=colors_p)
        for i, v in enumerate(probe_df[col]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
        ax.set_ylabel("Accuracy"); ax.set_title(title); ax.set_ylim(0, 1.1)
    plt.suptitle("Linear Probe (lower = better causal invariance)")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "metadata_probe.png"), dpi=150)
    plt.close()

    # --- 8B: Shuffle probe ---
    print("\n--- Shuffle Test: Probe on shuffled metadata labels ---")
    np.random.seed(123)
    sexes_shuffled = np.random.permutation(sexes_val)
    locs_shuffled  = np.random.permutation(locs_val)

    shuffle_results = []
    for idx_feat, (feat_name, feats) in enumerate(
            [("Image-Only", feats_img_val), ("Causal (Z_c)", z_c_val)]):
        clf_sex_s = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        clf_sex_s.fit(feats, sexes_shuffled)
        sex_s = accuracy_score(sexes_shuffled, clf_sex_s.predict(feats))

        clf_loc_s = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        clf_loc_s.fit(feats, locs_shuffled)
        loc_s = accuracy_score(locs_shuffled, clf_loc_s.predict(feats))

        orig_sex = probe_results[idx_feat]["Sex Probe Acc"]
        orig_loc = probe_results[idx_feat]["Loc Probe Acc"]
        shuffle_results.append({
            "Model": feat_name,
            "Sex (orig)": orig_sex, "Sex (shuf)": sex_s,
            "Sex Drop": orig_sex - sex_s,
            "Loc (orig)": orig_loc, "Loc (shuf)": loc_s,
            "Loc Drop": orig_loc - loc_s,
        })
        print(f"  {feat_name}:")
        print(f"    Sex  orig={orig_sex:.4f}  shuf={sex_s:.4f}  "
              f"drop={orig_sex - sex_s:.4f}")
        print(f"    Loc  orig={orig_loc:.4f}  shuf={loc_s:.4f}  "
              f"drop={orig_loc - loc_s:.4f}")

    shuffle_df = pd.DataFrame(shuffle_results)
    shuffle_df.to_csv(os.path.join(SAVE_DIR, "shuffle_probe.csv"), index=False)

    # =========================================================
    # STEP 9: OOD Test — Age < 50 vs Age >= 50
    #   Causal model should be more robust across age groups
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 9: OOD TEST — Age < 50 vs Age >= 50")
    print("=" * 60)

    ages_real_val = val_data["ages"].numpy() * 85.0
    mask_young = ages_real_val < 50
    mask_old   = ages_real_val >= 50

    print(f"  Val split: age<50 = {mask_young.sum()},  age>=50 = {mask_old.sum()}")

    ood_rows = []
    for model_name, preds_arr, labels_arr in [
        ("Image-Only", preds_i, labels_i),
        ("Causal-Factorized", preds_c, labels_c),
    ]:
        acc_young = (preds_arr[mask_young] == labels_arr[mask_young]).mean()
        acc_old   = (preds_arr[mask_old]   == labels_arr[mask_old]).mean()
        gap = abs(acc_young - acc_old)
        ood_rows.append({
            "Model": model_name,
            "Acc (age<50)": acc_young, "Acc (age>=50)": acc_old,
            "Gap": gap,
        })
        print(f"  {model_name}:  young={acc_young:.4f}  old={acc_old:.4f}  "
              f"gap={gap:.4f}")

    ood_df = pd.DataFrame(ood_rows)
    ood_df.to_csv(os.path.join(SAVE_DIR, "ood_age_test.csv"), index=False)

    # OOD bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(ood_df))
    w = 0.3
    ax.bar(x - w/2, ood_df["Acc (age<50)"], w, label="age < 50", color="#4C72B0")
    ax.bar(x + w/2, ood_df["Acc (age>=50)"], w, label="age >= 50", color="#C44E52")
    for i in range(len(ood_df)):
        ax.text(i - w/2, ood_df["Acc (age<50)"].iloc[i] + 0.005,
                f"{ood_df['Acc (age<50)'].iloc[i]:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, ood_df["Acc (age>=50)"].iloc[i] + 0.005,
                f"{ood_df['Acc (age>=50)'].iloc[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(ood_df["Model"])
    ax.set_ylabel("Disease Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("OOD Test: Disease Accuracy by Age Group")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ood_age_test.png"), dpi=150)
    plt.close()

    # =========================================================
    # STEP 10: Subgroup Analysis + Fairness
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 10: SUBGROUP ANALYSIS & FAIRNESS")
    print("=" * 60)

    sex_names_inv = {v: k for k, v in sex_map.items()}
    loc_names_inv = {v: k for k, v in loc_map.items()}

    subgroup_rows = []
    for model_name, preds_arr, labels_arr in [
        ("Image-Only", preds_i, labels_i),
        ("Causal-Factorized", preds_c, labels_c),
    ]:
        for sex_idx in sorted(sex_map.values()):
            mask = sexes_val == sex_idx
            if mask.sum() == 0:
                continue
            sub_acc = (preds_arr[mask] == labels_arr[mask]).mean()
            subgroup_rows.append({
                "Model": model_name, "Type": "Sex",
                "Subgroup": sex_names_inv[sex_idx],
                "N": int(mask.sum()), "Accuracy": sub_acc,
            })

        for loc_idx in sorted(loc_map.values()):
            mask = locs_val == loc_idx
            if mask.sum() < 5:
                continue
            sub_acc = (preds_arr[mask] == labels_arr[mask]).mean()
            subgroup_rows.append({
                "Model": model_name, "Type": "Localization",
                "Subgroup": loc_names_inv[loc_idx],
                "N": int(mask.sum()), "Accuracy": sub_acc,
            })

    subgroup_df = pd.DataFrame(subgroup_rows)
    subgroup_df.to_csv(os.path.join(SAVE_DIR, "subgroup_analysis.csv"),
                       index=False)

    # --- Subgroup bar chart: Sex ---
    sex_sub = subgroup_df[subgroup_df["Type"] == "Sex"]
    if not sex_sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sex_groups = sex_sub["Subgroup"].unique()
        x = np.arange(len(sex_groups)); w = 0.35
        for i, mdl in enumerate(["Image-Only", "Causal-Factorized"]):
            vals = [sex_sub[(sex_sub["Model"] == mdl) &
                            (sex_sub["Subgroup"] == s)]["Accuracy"].values
                    for s in sex_groups]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            bars = ax.bar(x + i * w, vals, w, label=mdl)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                        f"{v:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x + w / 2); ax.set_xticklabels(sex_groups)
        ax.set_ylabel("Disease Accuracy")
        ax.set_title("Disease Accuracy by Sex")
        ax.set_ylim(0, 1.1); ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "subgroup_sex.png"), dpi=150)
        plt.close()

    # --- Subgroup bar chart: Localization ---
    loc_sub = subgroup_df[subgroup_df["Type"] == "Localization"]
    if not loc_sub.empty:
        fig, ax = plt.subplots(figsize=(14, 6))
        loc_groups = sorted(loc_sub["Subgroup"].unique())
        x = np.arange(len(loc_groups)); w = 0.35
        for i, mdl in enumerate(["Image-Only", "Causal-Factorized"]):
            vals = [loc_sub[(loc_sub["Model"] == mdl) &
                            (loc_sub["Subgroup"] == s)]["Accuracy"].values
                    for s in loc_groups]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            ax.bar(x + i * w, vals, w, label=mdl)
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(loc_groups, rotation=45, ha="right")
        ax.set_ylabel("Disease Accuracy")
        ax.set_title("Disease Accuracy by Localization")
        ax.set_ylim(0, 1.1); ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "subgroup_localization.png"), dpi=150)
        plt.close()

    # --- Fairness metric: std of subgroup accuracies ---
    print("\n--- Fairness (std of subgroup accuracies, lower = better) ---")
    fairness_rows = []
    for mdl in ["Image-Only", "Causal-Factorized"]:
        sex_std = sex_sub[sex_sub["Model"] == mdl]["Accuracy"].std()
        loc_std = loc_sub[loc_sub["Model"] == mdl]["Accuracy"].std()
        fairness_rows.append({
            "Model": mdl, "Sex Acc Std": sex_std, "Loc Acc Std": loc_std,
        })
        print(f"  {mdl}: Sex std={sex_std:.4f}  Loc std={loc_std:.4f}")

    fairness_df = pd.DataFrame(fairness_rows)
    fairness_df.to_csv(os.path.join(SAVE_DIR, "fairness_metric.csv"),
                       index=False)

    # =========================================================
    # STEP 11: External Test — ISIC2018 Task3 GroundTruth
    #   True OOD evaluation on unseen test set (1511 images)
    #   No segmentation masks → center-crop preprocessing
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 11: EXTERNAL TEST — ISIC2018 Task3 GroundTruth (1511 images)")
    print("=" * 60)

    test_df = pd.read_csv(TEST_GT_CSV)
    test_df["age"] = test_df["age"].fillna(test_df["age"].mean())
    test_df["sex"] = test_df["sex"].fillna("unknown")
    test_df["localization"] = test_df["localization"].fillna("unknown")
    print(f"  External test set: {len(test_df)} samples")
    print(f"  dx distribution:\n{test_df['dx'].value_counts().to_string()}")

    # Preprocess test images (no segmentation → center-crop)
    test_cache = os.path.join(CACHE_DIR, "test_external_data.pt")
    test_data = preprocess_test_cache(
        test_df, TEST_IMAGE_DIR, dx_map, sex_map, loc_map,
        target_size=TARGET_SIZE, cache_path=test_cache,
    )

    test_img_ds = CachedImageDataset(test_data, transform=val_aug)
    test_img_loader = DataLoader(test_img_ds, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=0, pin_memory=True)

    test_meta_ds = CachedMetaDataset(test_data, num_sex, num_loc)
    test_meta_loader = DataLoader(test_meta_ds, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0, pin_memory=True)

    # --- 11A: Evaluate all 3 models on external test ---
    print("\n--- 11A: Model Accuracy on External Test Set ---")
    ext_results = {}

    # Image-Only
    preds_ext_i, probs_ext_i, labels_ext_i = evaluate_on_loader(
        model_img, test_img_loader, device, "image_only")
    acc_ext_i, cm_ext_i, ss_ext_i = compute_metrics(
        preds_ext_i, labels_ext_i, num_classes)
    ss_ext_i["class_name"] = [class_names_list[i] for i in ss_ext_i["class_idx"]]
    ext_results["Image-Only"] = {
        "acc": acc_ext_i, "cm": cm_ext_i, "ss": ss_ext_i,
        "preds": preds_ext_i, "probs": probs_ext_i, "labels": labels_ext_i}

    # Meta-Only
    preds_ext_m, probs_ext_m, labels_ext_m = evaluate_on_loader(
        model_meta, test_meta_loader, device, "meta_only")
    acc_ext_m, cm_ext_m, ss_ext_m = compute_metrics(
        preds_ext_m, labels_ext_m, num_classes)
    ss_ext_m["class_name"] = [class_names_list[i] for i in ss_ext_m["class_idx"]]
    ext_results["Meta-Only"] = {
        "acc": acc_ext_m, "cm": cm_ext_m, "ss": ss_ext_m,
        "preds": preds_ext_m, "probs": probs_ext_m, "labels": labels_ext_m}

    # Causal Factorized
    preds_ext_c, probs_ext_c, labels_ext_c = evaluate_on_loader(
        model_causal, test_img_loader, device, "causal_factorized")
    acc_ext_c, cm_ext_c, ss_ext_c = compute_metrics(
        preds_ext_c, labels_ext_c, num_classes)
    ss_ext_c["class_name"] = [class_names_list[i] for i in ss_ext_c["class_idx"]]
    ext_results["Causal-Factorized"] = {
        "acc": acc_ext_c, "cm": cm_ext_c, "ss": ss_ext_c,
        "preds": preds_ext_c, "probs": probs_ext_c, "labels": labels_ext_c}

    # Print summary
    ext_summary_rows = []
    for name in ext_results:
        r = ext_results[name]
        ext_summary_rows.append({
            "Model": name, "Ext Test Accuracy": r["acc"],
            "Mean Sensitivity": r["ss"]["sensitivity"].mean(),
            "Mean Specificity": r["ss"]["specificity"].mean(),
        })
        print(f"  {name}: Acc={r['acc']:.4f}")
    ext_summary_df = pd.DataFrame(ext_summary_rows)
    ext_summary_df.to_csv(os.path.join(SAVE_DIR, "ext_test_summary.csv"),
                          index=False)

    # --- 11B: Plots for external test ---
    for name, key in [("Image-Only", "image_only"),
                      ("Meta-Only", "meta_only"),
                      ("Causal-Factorized", "causal_factorized")]:
        r = ext_results[name]
        plot_confusion_matrix(r["cm"], class_names_list,
                              f"External Test CM — {name}",
                              os.path.join(SAVE_DIR, f"ext_cm_{key}.png"))
        plot_roc(r["probs"], r["labels"], num_classes, class_names_list,
                 f"External Test ROC — {name}",
                 os.path.join(SAVE_DIR, f"ext_roc_{key}.png"))
        r["ss"].to_csv(os.path.join(SAVE_DIR, f"ext_sens_spec_{key}.csv"),
                       index=False)

    # External test accuracy bar chart
    ext_names = list(ext_results.keys())
    ext_accs = [ext_results[n]["acc"] for n in ext_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(ext_names, ext_accs, color=["#4C72B0", "#55A868", "#C44E52"])
    for bar, a in zip(bars, ext_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{a:.4f}", ha="center", fontweight="bold")
    plt.ylabel("Accuracy")
    plt.title("External Test (ISIC2018 Task3) — Accuracy")
    plt.ylim(0, 1.05); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ext_comparison_accuracy.png"), dpi=150)
    plt.close()

    # Val vs External accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ext_names)); w = 0.3
    val_accs = [results[n]["acc"] for n in ext_names]
    ax.bar(x - w/2, val_accs, w, label="Val (HAM10000)", color="#4C72B0")
    ax.bar(x + w/2, ext_accs, w, label="Ext Test (ISIC2018)", color="#C44E52")
    for i in range(len(ext_names)):
        ax.text(i - w/2, val_accs[i] + 0.005,
                f"{val_accs[i]:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, ext_accs[i] + 0.005,
                f"{ext_accs[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(ext_names)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Val vs External Test Accuracy")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "val_vs_ext_accuracy.png"), dpi=150)
    plt.close()

    # --- 11C: External test subgroup analysis ---
    print("\n--- 11C: External Test Subgroup Analysis ---")
    sexes_ext = test_data["sexes"].numpy()
    locs_ext  = test_data["locs"].numpy()

    ext_subgroup_rows = []
    for model_name, preds_arr, labels_arr in [
        ("Image-Only", preds_ext_i, labels_ext_i),
        ("Causal-Factorized", preds_ext_c, labels_ext_c),
    ]:
        for sex_idx in sorted(sex_map.values()):
            mask = sexes_ext == sex_idx
            if mask.sum() == 0:
                continue
            sub_acc = (preds_arr[mask] == labels_arr[mask]).mean()
            ext_subgroup_rows.append({
                "Model": model_name, "Type": "Sex",
                "Subgroup": sex_names_inv[sex_idx],
                "N": int(mask.sum()), "Accuracy": sub_acc,
            })

        for loc_idx in sorted(loc_map.values()):
            mask = locs_ext == loc_idx
            if mask.sum() < 5:
                continue
            sub_acc = (preds_arr[mask] == labels_arr[mask]).mean()
            ext_subgroup_rows.append({
                "Model": model_name, "Type": "Localization",
                "Subgroup": loc_names_inv[loc_idx],
                "N": int(mask.sum()), "Accuracy": sub_acc,
            })

    ext_subgroup_df = pd.DataFrame(ext_subgroup_rows)
    ext_subgroup_df.to_csv(
        os.path.join(SAVE_DIR, "ext_subgroup_analysis.csv"), index=False)

    # External fairness
    ext_sex_sub = ext_subgroup_df[ext_subgroup_df["Type"] == "Sex"]
    ext_loc_sub = ext_subgroup_df[ext_subgroup_df["Type"] == "Localization"]

    print("\n--- External Test Fairness ---")
    ext_fairness_rows = []
    for mdl in ["Image-Only", "Causal-Factorized"]:
        sex_std = ext_sex_sub[ext_sex_sub["Model"] == mdl]["Accuracy"].std()
        loc_std = ext_loc_sub[ext_loc_sub["Model"] == mdl]["Accuracy"].std()
        ext_fairness_rows.append({
            "Model": mdl, "Sex Acc Std": sex_std, "Loc Acc Std": loc_std,
        })
        print(f"  {mdl}: Sex std={sex_std:.4f}  Loc std={loc_std:.4f}")

    ext_fairness_df = pd.DataFrame(ext_fairness_rows)
    ext_fairness_df.to_csv(
        os.path.join(SAVE_DIR, "ext_fairness_metric.csv"), index=False)

    # --- 11D: External OOD by age ---
    ages_ext_real = test_data["ages"].numpy() * 85.0
    mask_ext_young = ages_ext_real < 50
    mask_ext_old   = ages_ext_real >= 50

    print(f"\n--- External OOD: age<50={mask_ext_young.sum()}, "
          f"age>=50={mask_ext_old.sum()} ---")
    ext_ood_rows = []
    for model_name, preds_arr, labels_arr in [
        ("Image-Only", preds_ext_i, labels_ext_i),
        ("Causal-Factorized", preds_ext_c, labels_ext_c),
    ]:
        acc_y = (preds_arr[mask_ext_young] == labels_arr[mask_ext_young]).mean()
        acc_o = (preds_arr[mask_ext_old]   == labels_arr[mask_ext_old]).mean()
        gap = abs(acc_y - acc_o)
        ext_ood_rows.append({
            "Model": model_name,
            "Acc (age<50)": acc_y, "Acc (age>=50)": acc_o, "Gap": gap,
        })
        print(f"  {model_name}: young={acc_y:.4f}  old={acc_o:.4f}  gap={gap:.4f}")

    ext_ood_df = pd.DataFrame(ext_ood_rows)
    ext_ood_df.to_csv(os.path.join(SAVE_DIR, "ext_ood_age_test.csv"),
                      index=False)

    # =========================================================
    # STEP 11.5: External Test — KNOWN-ONLY subset  [C.7]
    #   Drop rows whose sex or localization is "unknown" on the
    #   ISIC 2018 Task 3 test set. The bulk of the apparent val→ext
    #   epidemiological shift in v1 was driven by the high "unknown"
    #   rate (a labeling artifact), not by a genuine population shift.
    #   Reporting both versions makes the artifact explicit.
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 11.5: EXTERNAL TEST — known-only subset (C.7)")
    print("=" * 60)

    unk_sex_idx = sex_map["unknown"]
    unk_loc_idx = loc_map.get("unknown", -1)
    known_mask = (sexes_ext != unk_sex_idx) & (locs_ext != unk_loc_idx)
    n_known = int(known_mask.sum())
    n_total = int(known_mask.shape[0])
    print(f"  Known-only subset: {n_known} / {n_total} "
          f"({100.0 * n_known / n_total:.1f}%)")
    print(f"  Dropped: sex==unknown OR localization==unknown")

    if n_known < 20:
        print("  WARNING: known-only subset too small; skipping STEP 11.5")
    else:
        # ----- 11.5A: Per-model summary on filtered subset -----
        ext_known_results = {}
        ext_known_summary_rows = []
        for name in ext_results:
            r = ext_results[name]
            preds_k  = r["preds"][known_mask]
            probs_k  = r["probs"][known_mask]
            labels_k = r["labels"][known_mask]
            acc_k, cm_k, ss_k = compute_metrics(preds_k, labels_k, num_classes)
            ss_k["class_name"] = [class_names_list[i] for i in ss_k["class_idx"]]
            ext_known_results[name] = {
                "acc": acc_k, "cm": cm_k, "ss": ss_k,
                "preds": preds_k, "probs": probs_k, "labels": labels_k,
            }
            ext_known_summary_rows.append({
                "Model": name, "N": n_known,
                "Ext Test Accuracy (known-only)": acc_k,
                "Mean Sensitivity (known-only)": ss_k["sensitivity"].mean(),
                "Mean Specificity (known-only)": ss_k["specificity"].mean(),
            })
            print(f"  {name}: Acc(known)={acc_k:.4f}  "
                  f"vs Acc(full)={ext_results[name]['acc']:.4f}")

        ext_known_summary_df = pd.DataFrame(ext_known_summary_rows)
        ext_known_summary_df.to_csv(
            os.path.join(SAVE_DIR, "ext_test_summary_no_unknown.csv"),
            index=False)

        # ----- 11.5B: Confusion matrices on filtered subset -----
        for name, key in [("Image-Only", "image_only"),
                          ("Meta-Only", "meta_only"),
                          ("Causal-Factorized", "causal_factorized")]:
            r = ext_known_results[name]
            plot_confusion_matrix(
                r["cm"], class_names_list,
                f"External Test (known-only) CM — {name}",
                os.path.join(SAVE_DIR, f"ext_known_cm_{key}.png"))
            r["ss"].to_csv(
                os.path.join(SAVE_DIR, f"ext_known_sens_spec_{key}.csv"),
                index=False)

        # ----- 11.5C: Full vs known-only accuracy bar chart -----
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(ext_results.keys())
        x = np.arange(len(names)); w = 0.3
        full_accs  = [ext_results[n]["acc"] for n in names]
        known_accs = [ext_known_results[n]["acc"] for n in names]
        ax.bar(x - w/2, full_accs,  w, label=f"Full (N={n_total})",
               color="#C44E52")
        ax.bar(x + w/2, known_accs, w, label=f"Known-only (N={n_known})",
               color="#4C72B0")
        for i in range(len(names)):
            ax.text(i - w/2, full_accs[i] + 0.005,
                    f"{full_accs[i]:.3f}", ha="center", fontsize=9)
            ax.text(i + w/2, known_accs[i] + 0.005,
                    f"{known_accs[i]:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
        ax.set_title("External Test: Full vs Known-Only Subset (C.7)")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "ext_full_vs_known_accuracy.png"),
                    dpi=150)
        plt.close()

        # ----- 11.5D: Subgroup analysis on known-only -----
        sexes_ext_k = sexes_ext[known_mask]
        locs_ext_k  = locs_ext[known_mask]
        ext_known_subgroup_rows = []
        for model_name in ["Image-Only", "Causal-Factorized"]:
            preds_arr  = ext_known_results[model_name]["preds"]
            labels_arr = ext_known_results[model_name]["labels"]
            for sex_idx in sorted(sex_map.values()):
                if sex_idx == unk_sex_idx:
                    continue
                mask = sexes_ext_k == sex_idx
                if mask.sum() == 0:
                    continue
                sub_acc = (preds_arr[mask] == labels_arr[mask]).mean()
                ext_known_subgroup_rows.append({
                    "Model": model_name, "Type": "Sex",
                    "Subgroup": sex_names_inv[sex_idx],
                    "N": int(mask.sum()), "Accuracy": sub_acc,
                })
            for loc_idx in sorted(loc_map.values()):
                if loc_idx == unk_loc_idx:
                    continue
                mask = locs_ext_k == loc_idx
                if mask.sum() < 5:
                    continue
                sub_acc = (preds_arr[mask] == labels_arr[mask]).mean()
                ext_known_subgroup_rows.append({
                    "Model": model_name, "Type": "Localization",
                    "Subgroup": loc_names_inv[loc_idx],
                    "N": int(mask.sum()), "Accuracy": sub_acc,
                })
        ext_known_subgroup_df = pd.DataFrame(ext_known_subgroup_rows)
        ext_known_subgroup_df.to_csv(
            os.path.join(SAVE_DIR, "ext_known_subgroup_analysis.csv"),
            index=False)

        # ----- 11.5E: Fairness on known-only -----
        ext_known_fairness_rows = []
        for mdl in ["Image-Only", "Causal-Factorized"]:
            sub = ext_known_subgroup_df[ext_known_subgroup_df["Model"] == mdl]
            sex_std = sub[sub["Type"] == "Sex"]["Accuracy"].std()
            loc_std = sub[sub["Type"] == "Localization"]["Accuracy"].std()
            ext_known_fairness_rows.append({
                "Model": mdl, "Sex Acc Std": sex_std, "Loc Acc Std": loc_std,
            })
            print(f"  Fairness (known-only) {mdl}: "
                  f"Sex std={sex_std:.4f}  Loc std={loc_std:.4f}")
        pd.DataFrame(ext_known_fairness_rows).to_csv(
            os.path.join(SAVE_DIR, "ext_known_fairness_metric.csv"),
            index=False)

        # ----- 11.5F: OOD age on known-only -----
        ages_ext_real_k = ages_ext_real[known_mask]
        mask_yk = ages_ext_real_k < 50
        mask_ok = ages_ext_real_k >= 50
        print(f"  OOD age (known-only): "
              f"age<50={int(mask_yk.sum())}, age>=50={int(mask_ok.sum())}")
        ext_known_ood_rows = []
        for model_name in ["Image-Only", "Causal-Factorized"]:
            preds_arr  = ext_known_results[model_name]["preds"]
            labels_arr = ext_known_results[model_name]["labels"]
            acc_y = (preds_arr[mask_yk] == labels_arr[mask_yk]).mean() \
                if mask_yk.any() else float("nan")
            acc_o = (preds_arr[mask_ok] == labels_arr[mask_ok]).mean() \
                if mask_ok.any() else float("nan")
            gap = abs(acc_y - acc_o) if (mask_yk.any() and mask_ok.any()) \
                else float("nan")
            ext_known_ood_rows.append({
                "Model": model_name,
                "Acc (age<50)": acc_y, "Acc (age>=50)": acc_o, "Gap": gap,
            })
            print(f"  {model_name}: young={acc_y:.4f}  old={acc_o:.4f}  "
                  f"gap={gap:.4f}")
        pd.DataFrame(ext_known_ood_rows).to_csv(
            os.path.join(SAVE_DIR, "ext_known_ood_age_test.csv"), index=False)

        # ----- 11.5G: Drop attributable to "unknown" rows -----
        drop_rows = []
        for name in names:
            full_a  = ext_results[name]["acc"]
            known_a = ext_known_results[name]["acc"]
            drop_rows.append({
                "Model": name,
                "Acc (full)": full_a,
                "Acc (known-only)": known_a,
                "Delta (known - full)": known_a - full_a,
            })
        pd.DataFrame(drop_rows).to_csv(
            os.path.join(SAVE_DIR, "ext_unknown_artifact.csv"), index=False)
        print("\n  C.7 takeaway: if Delta > 0 across models, the val→ext gap")
        print("  in v1 was inflated by 'unknown' metadata rows (labeling")
        print("  artifact) rather than genuine epidemiological shift.")

    # =========================================================
    # STEP 12: Causal DAG — Interpretability Analysis
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 12: CAUSAL DAG — Interpretability Analysis")
    print("=" * 60)

    # 12A: Build and draw the causal DAG
    print("\n--- 12A: Causal DAG Structure ---")
    dag = build_causal_dag()
    print(f"  DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    for u, v, d in dag.edges(data=True):
        print(f"    {u} → {v}  [{d['edge_type']}] {d['desc']}")

    # 12B: Compute empirical edge strengths from data
    print("\n--- 12B: Empirical Edge Strengths (Cramér's V / Correlation Ratio) ---")
    edge_strengths = compute_edge_strengths(df, dx_map, sex_map, loc_map)
    for (u, v), val in sorted(edge_strengths.items()):
        print(f"    {u} → {v}: {val:.4f}")
    pd.DataFrame([{"Edge": f"{u}→{v}", "Strength": val}
                   for (u, v), val in edge_strengths.items()]).to_csv(
        os.path.join(SAVE_DIR, "dag_edge_strengths.csv"), index=False)

    # 12C: Plot the DAG with edge strengths
    print("\n--- 12C: Plotting causal DAG ---")
    plot_causal_dag(dag, os.path.join(SAVE_DIR, "causal_dag.png"),
                    edge_strengths=edge_strengths)
    print("  Saved causal_dag.png")

    # 12D: Intervention summary DAG
    plot_dag_intervention_summary(
        os.path.join(SAVE_DIR, "causal_dag_interventions.png"))
    print("  Saved causal_dag_interventions.png")

    # 12E: Conditional probability tables & heatmaps
    print("\n--- 12E: Conditional Probability Tables ---")
    cond_probs = compute_conditional_probs(df, dx_map, sex_map, loc_map,
                                           SAVE_DIR)
    plot_conditional_heatmaps(cond_probs, SAVE_DIR)
    for name, table in cond_probs.items():
        print(f"\n  {name}:")
        print(table.to_string())

    # 12F: Model-DAG alignment table
    print("\n--- 12F: Model-DAG Alignment ---")
    final_orth = hist_causal["L_orth"][-1] if hist_causal["L_orth"] else 0
    final_minv = hist_causal["L_minv"][-1] if hist_causal["L_minv"] else 0

    plot_model_dag_alignment(
        probe_results, shuffle_results,
        orth_loss=final_orth, minv_loss=final_minv,
        save_path=os.path.join(SAVE_DIR, "model_dag_alignment.png"),
    )
    print("  Saved model_dag_alignment.png")

    # 12G: Summary of causal analysis
    print("\n--- 12G: Causal Interpretation Summary ---")
    print("  Causal DAG encodes medical prior knowledge:")
    print("    - Disease (Y) CAUSES lesion morphology in image (X)")
    print("    - Age, Sex, Localization are CONFOUNDERS (affect both Y and X)")
    print("  Our model addresses confounding via:")
    print(f"    1. L_orth={final_orth:.4f}: forces Z_c ⊥ Z_s")
    print(f"    2. L_minv={final_minv:.4f}: prediction invariant to metadata shuffle")
    print(f"       (marginal independence; NOT a true counterfactual on do(M))")
    print(f"    3. L_IRM: classifier optimal across age environments")
    print(f"    4. C.2 architecture: Z_c receives image only, no metadata input")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY — Causal Representation Learning v3")
    print("=" * 60)

    print("\n[1] Model Accuracy — Validation (HAM10000)")
    print(summary_df.to_string(index=False))

    print("\n[2] Model Accuracy — External Test (ISIC2018 Task3)")
    print(ext_summary_df.to_string(index=False))

    print("\n[3] Linear Probe: Metadata from features (lower = better)")
    print(probe_df.to_string(index=False))
    print("  -> Causal Z_c should be near Random baseline")

    print("\n[4] Shuffle Test: Original vs Shuffled probe")
    print(shuffle_df.to_string(index=False))
    print("  -> Image-Only: large drop = features encode real metadata")
    print("  -> Causal Z_c: small drop = features independent of metadata")

    print("\n[5] OOD Test (Val): Age < 50 vs Age >= 50")
    print(ood_df.to_string(index=False))

    print("\n[6] OOD Test (Ext): Age < 50 vs Age >= 50")
    print(ext_ood_df.to_string(index=False))
    print("  -> Causal should have smaller gap (IRM enforces invariance)")

    print("\n[7] Fairness — Val")
    print(fairness_df.to_string(index=False))

    print("\n[8] Fairness — External Test")
    print(ext_fairness_df.to_string(index=False))
    print("  -> Lower std = more consistent across subgroups")

    print("\n[9] Causal DAG Edge Strengths (data-driven)")
    for (u, v), val in sorted(edge_strengths.items()):
        print(f"    {u} → {v}: {val:.4f}")
    print("  -> L→Y strongest confounder = localization predicts disease")
    print("  -> Model must block confounding paths A→X, S→X, L→X")

    print(f"\nAll results saved to {SAVE_DIR}/")
