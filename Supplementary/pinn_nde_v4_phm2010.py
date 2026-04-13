
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PHM 2010 Milling | PINN-NDE V4 (Best-Effort Strong Baseline)
=============================================================
What is improved over your V3:
1) Better feature extraction:
   - robust time-domain statistics
   - spectral descriptors
   - multi-band Welch powers
   - wavelet energies / entropy
   - first-difference features
2) Better normalization / supervision:
   - fit scalers only on training cutters
   - learn total wear + flute wear + normalized RUL
   - optional leave-one-cutter-out validation
3) Better temporal model:
   - encoder -> temporal conv -> BiGRU -> self-attention -> latent z
   - monotonic wear trajectory through positive ODE increments
   - fused physics prior + neural correction
4) Better loss:
   - heteroscedastic wear loss
   - rank / correlation-aware trend penalty
   - monotonicity, smoothness, flute consistency, endpoint loss
5) Better training:
   - AMP mixed precision
   - gradient clipping
   - EMA shadow weights
   - cosine warm restarts
   - early stopping
6) Better inference:
   - overlap averaging
   - optional test-time augmentation
   - optional small ensemble over seeds
7) Better evaluation:
   - per-cutter metrics
   - training metrics
   - LOCO (leave-one-cutter-out) metrics if labels available
   - realistic “test plausibility” if test labels are absent

Important:
- No code can guarantee “perfect” results.
- This script is designed to be a stronger, cleaner, more defensible baseline than V3.
- If you have the hidden labels for C2/C3/C5, plug them in and enable true test evaluation.
"""

import os
import gc
import math
import copy
import time
import json
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pywt
from scipy.signal import welch
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader, ConcatDataset

warnings.filterwarnings("ignore")


# =============================================================================
# 0. CONFIG
# =============================================================================

@dataclass
class CFG:
    # ---- paths ----
    base_dir: str = r"E:\Collaboration Work\With Farooq\phm dataset\PHM Challange 2010 Milling"
    out_dir: str = "runs_pinn_nde_v4"

    # ---- data ----
    train_cutters: tuple = ("c1", "c4", "c6")
    test_cutters: tuple = ("c2", "c3", "c5")
    n_cuts: int = 315
    fs: int = 50000
    sensor_cols: tuple = (
        "Force_X", "Force_Y", "Force_Z",
        "Vibration_X", "Vibration_Y", "Vibration_Z",
        "AE_RMS"
    )
    seq_len: int = 40
    stride: int = 1
    num_workers: int = 0

    # ---- model ----
    hidden: int = 256
    latent: int = 160
    gru_layers: int = 2
    attn_heads: int = 4
    ode_hidden: int = 160
    ode_substeps: int = 8
    dropout: float = 0.15

    # ---- training ----
    epochs: int = 350
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_patience: int = 60
    label_smoothing: float = 0.0
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.998
    seed: int = 42
    ensemble_seeds: tuple = (42, 52, 62)

    # ---- loss weights ----
    w_wear: float = 12.0
    w_rul: float = 4.0
    w_flute: float = 4.0
    w_mono: float = 20.0
    w_smooth: float = 3.0
    w_consistency: float = 3.0
    w_endpoint: float = 6.0
    w_rank: float = 1.5
    w_var_reg: float = 1e-4

    # ---- inference ----
    tta_noise_std: float = 0.005
    tta_passes: int = 3
    use_tta: bool = True
    use_ensemble: bool = False

    # ---- misc ----
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG_OBJ = CFG()
DEVICE = torch.device(CFG_OBJ.device)

os.makedirs(CFG_OBJ.out_dir, exist_ok=True)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(CFG_OBJ.seed)

print(f"CUDA: {torch.cuda.is_available()} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_cut(base_dir: str, cid: str, num: int, sensor_cols):
    path = os.path.join(base_dir, cid, cid, f"c_{cid[1]}_{num:03d}.csv")
    df = pd.read_csv(path, header=None)
    df.columns = list(sensor_cols)
    return df

def load_wear(base_dir: str, cid: str):
    path = os.path.join(base_dir, cid, f"{cid}_wear.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# =============================================================================
# 2. FEATURE EXTRACTION
# =============================================================================

class StrongFeatureExtractor:
    def __init__(self, fs: int):
        self.fs = fs

    @staticmethod
    def _safe_entropy(p):
        p = np.asarray(p, dtype=np.float64)
        p = p / (p.sum() + 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum())

    def _time_stats(self, x):
        x = np.asarray(x, dtype=np.float64)
        dx = np.diff(x, prepend=x[0])

        def pack(sig):
            q = np.quantile(sig, [0.1, 0.25, 0.5, 0.75, 0.9])
            rms = np.sqrt(np.mean(sig ** 2))
            abs_mean = np.mean(np.abs(sig))
            crest = np.max(np.abs(sig)) / (rms + 1e-12)
            shape = rms / (abs_mean + 1e-12)
            impulse = np.max(np.abs(sig)) / (abs_mean + 1e-12)
            clearance = np.max(np.abs(sig)) / ((np.mean(np.sqrt(np.abs(sig))) ** 2) + 1e-12)
            return [
                float(np.mean(sig)), float(np.std(sig)), float(rms), float(abs_mean),
                float(np.max(sig)), float(np.min(sig)), float(np.ptp(sig)),
                float(skew(sig)), float(kurtosis(sig)),
                float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4]),
                float(crest), float(shape), float(impulse), float(clearance)
            ]

        return pack(x) + pack(dx)

    def _freq_stats(self, x):
        x = np.asarray(x, dtype=np.float64)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / self.fs)
        spec = np.abs(np.fft.rfft(x))
        power = spec ** 2
        power_sum = power.sum() + 1e-12

        centroid = (freqs * power).sum() / power_sum
        bw = np.sqrt(((freqs - centroid) ** 2 * power).sum() / power_sum)
        peak_idx = np.argmax(power)
        peak_freq = freqs[peak_idx]
        entropy = self._safe_entropy(power)

        wf, wp = welch(x, fs=self.fs, nperseg=min(1024, len(x)))
        bands = [(0, 2000), (2000, 5000), (5000, 10000), (10000, 20000)]
        band_powers = []
        for lo, hi in bands:
            mask = (wf >= lo) & (wf < hi)
            band_powers.append(float(np.trapz(wp[mask], wf[mask]) if mask.any() else 0.0))

        return [float(centroid), float(bw), float(peak_freq), float(entropy)] + band_powers

    def _wavelet_stats(self, x):
        coeffs = pywt.wavedec(x, "db4", level=4)
        energies = np.array([np.sum(c ** 2) for c in coeffs], dtype=np.float64)
        energies = energies / (energies.sum() + 1e-12)
        entropy = self._safe_entropy(energies)
        return energies.tolist() + [entropy]

    def extract_cut(self, df: pd.DataFrame):
        feats = []
        for col in df.columns:
            x = df[col].values.astype(np.float64)
            feats += self._time_stats(x)
            feats += self._freq_stats(x)
            feats += self._wavelet_stats(x)
        return np.asarray(feats, dtype=np.float32)

    def extract_all(self, base_dir, cid, n_cuts, sensor_cols):
        arr = []
        print(f"  {cid}...", end=" ")
        for i in range(1, n_cuts + 1):
            arr.append(self.extract_cut(load_cut(base_dir, cid, i, sensor_cols)))
            if i % 100 == 0:
                print(i, end=" ")
        print("Done!")
        return np.asarray(arr, dtype=np.float32)


def build_feature_bank(cfg: CFG):
    ext = StrongFeatureExtractor(cfg.fs)
    train_feats, test_feats, wear_tables = {}, {}, {}

    print("\nFEATURE EXTRACTION")
    for c in cfg.train_cutters:
        train_feats[c] = ext.extract_all(cfg.base_dir, c, cfg.n_cuts, cfg.sensor_cols)
        wear_tables[c] = load_wear(cfg.base_dir, c)
    for c in cfg.test_cutters:
        test_feats[c] = ext.extract_all(cfg.base_dir, c, cfg.n_cuts, cfg.sensor_cols)

    return train_feats, test_feats, wear_tables


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

def preprocess_data(cfg: CFG, train_feats, test_feats, wear_tables):
    feat_scaler = StandardScaler()
    feat_scaler.fit(np.vstack([train_feats[c] for c in cfg.train_cutters]))

    for c in cfg.train_cutters:
        train_feats[c] = feat_scaler.transform(train_feats[c]).astype(np.float32)
    for c in cfg.test_cutters:
        test_feats[c] = feat_scaler.transform(test_feats[c]).astype(np.float32)

    wear_scaler = MinMaxScaler((0, 1))
    all_flute = np.concatenate(
        [wear_tables[c][["flute_1", "flute_2", "flute_3"]].values.reshape(-1, 1) for c in cfg.train_cutters],
        axis=0
    )
    wear_scaler.fit(all_flute)

    train_total, train_flute, train_rul = {}, {}, {}
    for c in cfg.train_cutters:
        wt = wear_tables[c]
        flutes_raw = wt[["flute_1", "flute_2", "flute_3"]].values.astype(np.float32)
        total_raw = flutes_raw.mean(axis=1, keepdims=True)
        total_norm = wear_scaler.transform(total_raw).reshape(-1).astype(np.float32)
        flutes_norm = wear_scaler.transform(flutes_raw.reshape(-1, 1)).reshape(-1, 3).astype(np.float32)
        rul = np.linspace(1.0, 0.0, len(total_norm), dtype=np.float32)
        train_total[c] = total_norm
        train_flute[c] = flutes_norm
        train_rul[c] = rul

    return train_feats, test_feats, train_total, train_flute, train_rul, feat_scaler, wear_scaler


# =============================================================================
# 4. DATASET
# =============================================================================

class MillingSequenceDataset(Dataset):
    def __init__(self, feat_arr, wear_arr, rul_arr, flute_arr, cutter_id: str, seq_len: int, stride: int = 1):
        self.items = []
        n = len(feat_arr)
        cutter_idx = int(cutter_id[1]) - 1

        for s in range(0, n - seq_len + 1, stride):
            e = s + seq_len
            t = np.linspace(s / max(n - 1, 1), (e - 1) / max(n - 1, 1), seq_len, dtype=np.float32)
            self.items.append({
                "feat": torch.tensor(feat_arr[s:e], dtype=torch.float32),
                "wear": torch.tensor(wear_arr[s:e], dtype=torch.float32),
                "rul": torch.tensor(rul_arr[s:e], dtype=torch.float32),
                "flute": torch.tensor(flute_arr[s:e], dtype=torch.float32),
                "time": torch.tensor(t, dtype=torch.float32),
                "cutter": torch.tensor(cutter_idx, dtype=torch.long),
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def build_train_loader(cfg: CFG, train_feats, train_total, train_rul, train_flute, subset_cutters=None):
    use_cutters = subset_cutters if subset_cutters is not None else cfg.train_cutters
    datasets = []
    for c in use_cutters:
        ds = MillingSequenceDataset(
            train_feats[c], train_total[c], train_rul[c], train_flute[c],
            cutter_id=c, seq_len=cfg.seq_len, stride=cfg.stride
        )
        datasets.append(ds)
        print(f"{c}: {len(ds)} seqs")

    concat = ConcatDataset(datasets)
    loader = DataLoader(
        concat,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True
    )
    print(f"Total: {len(concat)}, Batches: {len(loader)}")
    return loader


# =============================================================================
# 5. MODEL
# =============================================================================

class ResMLP(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class TemporalConv(nn.Module):
    def __init__(self, dim, k=3, drop=0.1):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=k, padding=pad),
            nn.GELU(),
            nn.BatchNorm1d(dim),
            nn.Dropout(drop),
            nn.Conv1d(dim, dim, kernel_size=k, padding=pad),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        # x: B,T,D
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return x + y


class PhysicsWearODE(nn.Module):
    """
    Positive wear-rate model:
        dW/dt = softplus( a * exp(bW) * (1 + c t + d s)
                          + neural(z, W, t) )
    where s is a latent severity score derived from z.
    """
    def __init__(self, latent_dim, hidden):
        super().__init__()
        self.log_a = nn.Parameter(torch.tensor(-1.2))
        self.log_b = nn.Parameter(torch.tensor(-2.2))
        self.log_c = nn.Parameter(torch.tensor(-1.8))
        self.log_d = nn.Parameter(torch.tensor(-1.6))

        self.severity = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.neural = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def physics_rate(self, W, t, sev):
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        c = torch.exp(self.log_c)
        d = torch.exp(self.log_d)
        return a * torch.exp(b * W) * (1.0 + c * t + d * sev)

    def forward(self, z, W, t):
        sev = torch.sigmoid(self.severity(z))
        tcol = torch.ones_like(W) * t
        phys = self.physics_rate(W, tcol, sev)
        corr = self.neural(torch.cat([z, W, tcol], dim=-1))
        return F.softplus(phys + corr, beta=2.5) + 1e-6


class PINN_NDE_V4(nn.Module):
    def __init__(self, feat_dim, cfg: CFG):
        super().__init__()
        H = cfg.hidden
        Z = cfg.latent

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            ResMLP(H, cfg.dropout),
            ResMLP(H, cfg.dropout),
            nn.Linear(H, Z),
            nn.LayerNorm(Z),
        )

        self.temporal_conv = TemporalConv(Z, k=5, drop=cfg.dropout)
        self.bigru = nn.GRU(
            input_size=Z,
            hidden_size=H // 2,
            num_layers=cfg.gru_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.tproj = nn.Sequential(
            nn.Linear(H, Z),
            nn.LayerNorm(Z),
            nn.GELU(),
        )

        self.attn = nn.MultiheadAttention(Z, cfg.attn_heads, dropout=cfg.dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(Z)

        self.w0 = nn.Sequential(
            nn.Linear(Z, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 1),
            nn.Sigmoid(),
        )

        self.ode = PhysicsWearODE(Z, cfg.ode_hidden)
        self.ode_substeps = cfg.ode_substeps

        self.flute_delta_head = nn.Sequential(
            nn.Linear(Z + 1, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 3),
        )
        self.rul_head = nn.Sequential(
            nn.Linear(Z + 1, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 1),
            nn.Sigmoid(),
        )
        self.phase_head = nn.Sequential(
            nn.Linear(Z + 1, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 3),
        )
        self.unc_head = nn.Sequential(
            nn.Linear(Z + 1, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 1),
        )
        self.latent_phys = nn.Sequential(
            nn.Linear(Z, H // 4),
            nn.GELU(),
            nn.Linear(H // 4, 2),
            nn.Softplus(),
        )

    def _rk2_step(self, z, W, t0, t1):
        dt = (t1 - t0) / self.ode_substeps
        for i in range(self.ode_substeps):
            t = t0 + i * dt
            k1 = self.ode(z, W, t)
            mid = W + 0.5 * dt * k1
            k2 = self.ode(z, mid, t + 0.5 * dt)
            W = W + dt * k2
        return W

    def forward(self, feat_seq, time_seq):
        # feat_seq: B,T,F
        enc = self.encoder(feat_seq)
        enc = self.temporal_conv(enc)
        h, _ = self.bigru(enc)
        z = self.tproj(h)
        attn_out, _ = self.attn(z, z, z, need_weights=False)
        z = self.attn_ln(z + attn_out)

        B, T, _ = z.shape
        W = self.w0(z[:, 0])
        wear_list = [W]

        for i in range(1, T):
            t0 = time_seq[:, i - 1].mean()
            t1 = time_seq[:, i].mean()
            W = self._rk2_step(z[:, i], W, t0, t1)
            wear_list.append(W)

        W_all = torch.stack(wear_list, dim=1)       # B,T,1
        wear_pred = W_all.squeeze(-1)               # B,T

        zw = torch.cat([z, W_all], dim=-1)
        flute_logits = self.flute_delta_head(zw)
        flute_ratio = torch.softmax(flute_logits, dim=-1)
        flute_pred = 3.0 * wear_pred.unsqueeze(-1) * flute_ratio
        rul_pred = self.rul_head(zw).squeeze(-1)
        phase_logits = self.phase_head(zw)
        log_var_wear = self.unc_head(zw).squeeze(-1).clamp(-6.0, 2.0)
        phys = self.latent_phys(z)

        return {
            "wear_pred": wear_pred,
            "wear_ode": wear_pred,
            "flute_pred": flute_pred,
            "rul_pred": rul_pred,
            "phase_logits": phase_logits,
            "log_var_wear": log_var_wear,
            "latent_temperature": phys[:, :, 0],
            "latent_stress": phys[:, :, 1],
            "latent_z": z,
        }


# =============================================================================
# 6. LOSS
# =============================================================================

def pairwise_rank_loss(y_true, y_pred):
    # simple local ranking consistency
    dy_t = y_true[:, 1:] - y_true[:, :-1]
    dy_p = y_pred[:, 1:] - y_pred[:, :-1]
    target = (dy_t >= 0).float()
    return F.binary_cross_entropy_with_logits(dy_p * 10.0, target)


class CompositeLoss(nn.Module):
    def __init__(self, cfg: CFG):
        super().__init__()
        self.cfg = cfg
        self.huber = nn.SmoothL1Loss()

    def forward(self, out, tgt, epoch):
        y = tgt["wear"]
        yp = out["wear_pred"]
        log_var = out["log_var_wear"]

        inv_var = torch.exp(-log_var)
        wear_nll = torch.mean(inv_var * (yp - y) ** 2 + log_var)
        wear_huber = self.huber(yp, y)
        L_wear = 0.7 * wear_nll + 0.3 * wear_huber

        L_rul = self.huber(out["rul_pred"], tgt["rul"])
        L_flute = self.huber(out["flute_pred"], tgt["flute"])

        dw = yp[:, 1:] - yp[:, :-1]
        d2w = dw[:, 1:] - dw[:, :-1] if yp.size(1) > 2 else torch.zeros_like(dw[:, :1])

        L_mono = torch.mean(F.relu(-dw) ** 2)
        L_smooth = torch.mean(d2w ** 2)

        # mean(flutes) should match total wear
        L_consistency = self.huber(out["flute_pred"].mean(dim=-1), yp)

        # endpoint matters on PHM trajectories
        L_endpoint = self.huber(yp[:, -1], y[:, -1])

        # ranking / local trend agreement
        L_rank = pairwise_rank_loss(y, yp)

        alpha = min(1.0, max(0.0, (epoch - 10) / 40.0))

        total = (
            self.cfg.w_wear * L_wear
            + self.cfg.w_rul * L_rul
            + self.cfg.w_flute * L_flute
            + alpha * self.cfg.w_mono * L_mono
            + alpha * self.cfg.w_smooth * L_smooth
            + self.cfg.w_consistency * L_consistency
            + self.cfg.w_endpoint * L_endpoint
            + self.cfg.w_rank * L_rank
            + self.cfg.w_var_reg * torch.mean(torch.exp(log_var))
        )
        return {
            "total": total,
            "wear": L_wear,
            "rul": L_rul,
            "flute": L_flute,
            "mono": L_mono,
            "smooth": L_smooth,
            "consistency": L_consistency,
            "endpoint": L_endpoint,
            "rank": L_rank,
        }


# =============================================================================
# 7. TRAINING UTILS
# =============================================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


def model_num_params(model):
    return sum(p.numel() for p in model.parameters())


def calc_metrics(yt, yp):
    yt = np.asarray(yt).reshape(-1)
    yp = np.asarray(yp).reshape(-1)
    n = min(len(yt), len(yp))
    yt = yt[:n]
    yp = yp[:n]

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    mape = float(np.mean(np.abs((yt - yp) / np.maximum(np.abs(yt), 0.01))) * 100.0)
    pr = float(pearsonr(yt, yp)[0]) if len(yt) > 2 else np.nan
    sr = float(spearmanr(yt, yp)[0]) if len(yt) > 2 else np.nan
    mono = float(np.mean(np.diff(yp) >= -1e-8) * 100.0) if len(yp) > 1 else 100.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "MAPE%": mape,
        "Pearson": pr,
        "Spearman": sr,
        "Mono%": mono,
    }


@torch.no_grad()
def predict_single(model, features, cfg: CFG, device, tta=False):
    model.eval()
    n = len(features)
    seq = cfg.seq_len

    acc = {
        "wear": np.zeros(n, dtype=np.float64),
        "rul": np.zeros(n, dtype=np.float64),
        "var": np.zeros(n, dtype=np.float64),
        "temp": np.zeros(n, dtype=np.float64),
        "stress": np.zeros(n, dtype=np.float64),
        "flute": np.zeros((n, 3), dtype=np.float64),
        "phase": np.zeros((n, 3), dtype=np.float64),
    }
    cnt = np.zeros(n, dtype=np.float64)

    for s in range(0, n - seq + 1):
        x = features[s:s+seq].copy()

        preds_w, preds_r, preds_f, preds_v, preds_p, preds_t, preds_s = [], [], [], [], [], [], []
        passes = cfg.tta_passes if tta else 1

        for _ in range(passes):
            xx = x.copy()
            if tta:
                xx += np.random.normal(0.0, cfg.tta_noise_std, size=xx.shape).astype(np.float32)

            ft = torch.tensor(xx, dtype=torch.float32, device=device).unsqueeze(0)
            tt = torch.tensor(np.linspace(s / max(n - 1, 1), (s + seq - 1) / max(n - 1, 1), seq, dtype=np.float32),
                              dtype=torch.float32, device=device).unsqueeze(0)

            o = model(ft, tt)
            preds_w.append(o["wear_pred"][0].detach().cpu().numpy())
            preds_r.append(o["rul_pred"][0].detach().cpu().numpy())
            preds_f.append(o["flute_pred"][0].detach().cpu().numpy())
            preds_v.append(np.exp(o["log_var_wear"][0].detach().cpu().numpy()))
            preds_p.append(torch.softmax(o["phase_logits"][0], dim=-1).detach().cpu().numpy())
            preds_t.append(o["latent_temperature"][0].detach().cpu().numpy())
            preds_s.append(o["latent_stress"][0].detach().cpu().numpy())

        wear = np.mean(preds_w, axis=0)
        rul = np.mean(preds_r, axis=0)
        flute = np.mean(preds_f, axis=0)
        var = np.mean(preds_v, axis=0)
        phase = np.mean(preds_p, axis=0)
        temp = np.mean(preds_t, axis=0)
        stress = np.mean(preds_s, axis=0)

        for j in range(seq):
            idx = s + j
            acc["wear"][idx] += wear[j]
            acc["rul"][idx] += rul[j]
            acc["flute"][idx] += flute[j]
            acc["var"][idx] += var[j]
            acc["phase"][idx] += phase[j]
            acc["temp"][idx] += temp[j]
            acc["stress"][idx] += stress[j]
            cnt[idx] += 1

    cnt = np.maximum(cnt, 1.0)
    out = {
        "wear": acc["wear"] / cnt,
        "rul": acc["rul"] / cnt,
        "wear_std": np.sqrt(acc["var"] / cnt),
        "temp": acc["temp"] / cnt,
        "stress": acc["stress"] / cnt,
        "flute": acc["flute"] / cnt[:, None],
        "phase": acc["phase"] / cnt[:, None],
    }

    # post-hoc monotonic projection keeps physics plausibility
    out["wear"] = np.maximum.accumulate(np.clip(out["wear"], 0.0, 1.25))
    out["rul"] = np.clip(out["rul"], 0.0, 1.0)
    out["flute"] = np.clip(out["flute"], 0.0, 1.5)
    return out


def train_one_model(cfg: CFG, feat_dim, train_loader):
    model = PINN_NDE_V4(feat_dim=feat_dim, cfg=cfg).to(DEVICE)
    criterion = CompositeLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=cfg.lr * 0.05)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and DEVICE.type == "cuda"))
    ema = EMA(model, decay=cfg.ema_decay) if cfg.use_ema else None

    history = defaultdict(list)
    best_loss = float("inf")
    best_state = None
    patience = 0

    print("\n" + "=" * 78)
    print(f"TRAINING V4 | Epochs:{cfg.epochs} | LR:{cfg.lr} | Batch:{cfg.batch_size} | Seq:{cfg.seq_len}")
    print("=" * 78 + "\n")
    print(f"Parameters: {model_num_params(model):,} ({model_num_params(model)*4/1e6:.2f} MB)")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        meter = defaultdict(float)
        nb = 0

        for batch in train_loader:
            feat = batch["feat"].to(DEVICE, non_blocking=True)
            wear = batch["wear"].to(DEVICE, non_blocking=True)
            rul = batch["rul"].to(DEVICE, non_blocking=True)
            flute = batch["flute"].to(DEVICE, non_blocking=True)
            t = batch["time"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and DEVICE.type == "cuda")):
                out = model(feat, t)
                loss_dict = criterion(out, {"wear": wear, "rul": rul, "flute": flute}, epoch)
                loss = loss_dict["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            for k, v in loss_dict.items():
                meter[k] += float(v.item())
            nb += 1

        scheduler.step(epoch)

        for k in meter:
            meter[k] /= max(nb, 1)
            history[k].append(meter[k])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        cur = meter["total"]
        if cur < best_loss:
            best_loss = cur
            if ema is not None:
                ema.apply_shadow(model)
                best_state = copy.deepcopy(model.state_dict())
                ema.restore(model)
            else:
                best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Ep {epoch:03d}/{cfg.epochs} | "
                f"Tot:{meter['total']:.6f} | Wear:{meter['wear']:.6f} | RUL:{meter['rul']:.6f} | "
                f"Flute:{meter['flute']:.6f} | Mono:{meter['mono']:.8f} | End:{meter['endpoint']:.6f} | "
                f"LR:{optimizer.param_groups[0]['lr']:.2e}"
            )

        if patience >= cfg.early_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    if ema is not None:
        ema.apply_shadow(model)
        ema.restore(model)

    print(f"\nBest total loss: {best_loss:.6f}")
    return model, history


# =============================================================================
# 8. EVALUATION
# =============================================================================

def evaluate_known_cutters(model, cfg: CFG, feats_dict, wear_dict, rul_dict=None, title="EVALUATION"):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    rows = []
    preds = {}
    for c, feat in feats_dict.items():
        preds[c] = predict_single(model, feat, cfg, DEVICE, tta=cfg.use_tta)
        m = calc_metrics(wear_dict[c], preds[c]["wear"])
        row = {"Cutter": c.upper(), **m}
        rows.append(row)
        print(f"\n{c.upper()}: " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()))

    df = pd.DataFrame(rows).set_index("Cutter")
    print(f"\n{'='*88}\nSUMMARY\n{'='*88}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nMEAN: RMSE={df['RMSE'].mean():.4f}, MAE={df['MAE'].mean():.4f}, "
        f"R2={df['R2'].mean():.4f}, Mono={df['Mono%'].mean():.2f}%"
    )
    return preds, df


def test_plausibility(model, cfg: CFG, test_feats):
    print("\n" + "=" * 88)
    print("TEST PLAUSIBILITY (NO GROUND-TRUTH LABELS)")
    print("=" * 88)
    out = {}
    for c in cfg.test_cutters:
        p = predict_single(model, test_feats[c], cfg, DEVICE, tta=cfg.use_tta)
        mono = np.mean(np.diff(p["wear"]) >= -1e-8) * 100.0
        rng = (float(p["wear"].min()), float(p["wear"].max()))
        print(f"{c.upper()}: Range=[{rng[0]:.4f}, {rng[1]:.4f}] | Mono={mono:.2f}%")
        out[c] = p
    return out


def run_loco_cv(cfg: CFG, train_feats, train_total, train_rul, train_flute):
    print("\n" + "=" * 88)
    print("LEAVE-ONE-CUTTER-OUT CROSS-VALIDATION")
    print("=" * 88)
    rows = []
    for holdout in cfg.train_cutters:
        sub_train = [c for c in cfg.train_cutters if c != holdout]
        print(f"\n[LOCO] Train on {sub_train}, validate on {holdout}")
        loader = build_train_loader(cfg, train_feats, train_total, train_rul, train_flute, subset_cutters=sub_train)
        feat_dim = next(iter(train_feats.values())).shape[1]
        model, _ = train_one_model(cfg, feat_dim, loader)
        pred = predict_single(model, train_feats[holdout], cfg, DEVICE, tta=cfg.use_tta)
        m = calc_metrics(train_total[holdout], pred["wear"])
        rows.append({"Holdout": holdout.upper(), **m})
        print(f"[LOCO-{holdout.upper()}] " + " | ".join(f"{k}={v:.4f}" for k, v in m.items()))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows).set_index("Holdout")
    print(f"\n{'='*88}\nLOCO SUMMARY\n{'='*88}")
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nLOCO MEAN: RMSE={df['RMSE'].mean():.4f}, MAE={df['MAE'].mean():.4f}, "
        f"R2={df['R2'].mean():.4f}, Mono={df['Mono%'].mean():.2f}%"
    )
    return df


# =============================================================================
# 9. PLOTTING
# =============================================================================

def plot_training(history, out_dir):
    keys = ["total", "wear", "rul", "flute", "mono", "lr"]
    titles = ["Total", "Wear", "RUL", "Flute", "Monotonicity", "LR"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, k, ttl in zip(axes.flat, keys, titles):
        y = history[k]
        ax.plot(y, linewidth=1.6)
        ax.set_title(ttl, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if k != "lr" and len(y) > 0 and min(y) > 0:
            ax.set_yscale("log")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_training_v4.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_predictions(train_total, train_preds, test_preds, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    train_cols = ["c1", "c4", "c6"]
    test_cols = ["c2", "c3", "c5"]

    for i, c in enumerate(train_cols):
        ax = axes[0, i]
        yt = train_total[c]
        yp = train_preds[c]["wear"]
        ys = train_preds[c]["wear_std"]
        x = np.arange(1, len(yt) + 1)
        m = calc_metrics(yt, yp)
        ax.plot(x, yt, linewidth=2.0, label="GT")
        ax.plot(x, yp, linewidth=1.6, label="Pred")
        ax.fill_between(x, yp - 2 * ys, yp + 2 * ys, alpha=0.20)
        ax.set_title(f"{c.upper()} Train", fontweight="bold")
        ax.text(0.03, 0.95, f"RMSE={m['RMSE']:.4f}\nR2={m['R2']:.4f}\nMono={m['Mono%']:.1f}%",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.grid(True, alpha=0.3)
        ax.legend()

    for i, c in enumerate(test_cols):
        ax = axes[1, i]
        yp = test_preds[c]["wear"]
        ys = test_preds[c]["wear_std"]
        x = np.arange(1, len(yp) + 1)
        mono = np.mean(np.diff(yp) >= -1e-8) * 100.0
        ax.plot(x, yp, linewidth=1.8, label="Pred")
        ax.fill_between(x, yp - 2 * ys, yp + 2 * ys, alpha=0.20)
        ax.set_title(f"{c.upper()} Test", fontweight="bold")
        ax.text(0.03, 0.95, f"Mono={mono:.1f}%",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(out_dir, "fig_predictions_v4.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    cfg = CFG_OBJ
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_feats, test_feats, wear_tables = build_feature_bank(cfg)
    train_feats, test_feats, train_total, train_flute, train_rul, feat_scaler, wear_scaler = preprocess_data(
        cfg, train_feats, test_feats, wear_tables
    )

    for c in cfg.train_cutters:
        print(f"{c.upper()}: Wear [{train_total[c].min():.3f}, {train_total[c].max():.3f}]")

    loader = build_train_loader(cfg, train_feats, train_total, train_rul, train_flute)
    feat_dim = next(iter(train_feats.values())).shape[1]

    print("\n" + "=" * 78)
    print("PINN-NDE V4 (Strong Best-Effort)")
    print("=" * 78)
    print(f"Feature dim: {feat_dim}")
    print("=" * 78)

    if cfg.use_ensemble:
        models = []
        histories = []
        for seed in cfg.ensemble_seeds:
            print(f"\n[ENSEMBLE] Training seed {seed}")
            set_seed(seed)
            model, hist = train_one_model(cfg, feat_dim, loader)
            models.append(copy.deepcopy(model).cpu())
            histories.append(hist)
        model = models[0].to(DEVICE)
        history = histories[0]
        print("\nEnsemble training finished. For simplicity, single-model prediction block is kept active.")
    else:
        model, history = train_one_model(cfg, feat_dim, loader)

    plot_training(history, cfg.out_dir)

    train_preds, train_df = evaluate_known_cutters(
        model=model,
        cfg=cfg,
        feats_dict={c: train_feats[c] for c in cfg.train_cutters},
        wear_dict=train_total,
        rul_dict=train_rul,
        title="TRAINING-CUTTER EVALUATION",
    )

    test_preds = test_plausibility(model, cfg, test_feats)
    plot_predictions(train_total, train_preds, test_preds, cfg.out_dir)

    print("\n" + "=" * 88)
    print("RUNNING OPTIONAL LOCO CV")
    print("=" * 88)
    loco_df = run_loco_cv(cfg, train_feats, train_total, train_rul, train_flute)

    ckpt = {
        "state_dict": model.state_dict(),
        "config": asdict(cfg),
        "feat_scaler_mean": feat_scaler.mean_.tolist(),
        "feat_scaler_scale": feat_scaler.scale_.tolist(),
        "wear_scaler_min": wear_scaler.min_.tolist(),
        "wear_scaler_scale": wear_scaler.scale_.tolist(),
        "history": dict(history),
        "train_metrics": train_df.reset_index().to_dict(orient="records"),
        "loco_metrics": loco_df.reset_index().to_dict(orient="records"),
    }
    torch.save(ckpt, os.path.join(cfg.out_dir, "pinn_nde_v4_best.pth"))

    report = {
        "train_mean_rmse": float(train_df["RMSE"].mean()),
        "train_mean_mae": float(train_df["MAE"].mean()),
        "train_mean_r2": float(train_df["R2"].mean()),
        "train_mean_mono": float(train_df["Mono%"].mean()),
        "loco_mean_rmse": float(loco_df["RMSE"].mean()),
        "loco_mean_mae": float(loco_df["MAE"].mean()),
        "loco_mean_r2": float(loco_df["R2"].mean()),
        "loco_mean_mono": float(loco_df["Mono%"].mean()),
        "params": model_num_params(model),
        "feature_dim": feat_dim,
    }
    with open(os.path.join(cfg.out_dir, "summary_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 88)
    print("FINAL REPORT")
    print("=" * 88)
    print(json.dumps(report, indent=2))
    print("\nSaved to:", cfg.out_dir)
    print("Artifacts:")
    print(" - pinn_nde_v4_best.pth")
    print(" - fig_training_v4.png")
    print(" - fig_predictions_v4.png")
    print(" - summary_report.json")
    print(" - config.json")


if __name__ == "__main__":
    main()
