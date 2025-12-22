"""
使用 HuggingFace Trainer 训练 CrossPVT_T2T_MambaDINO 头（ViT-7B frozen 特征）。

相比自定义循环，简化为：
- Trainer 负责 epoch/日志/保存。
- Dataset 返回 dict，DataCollator 直接堆叠。
- forward 返回 loss/logits，compute_metrics 计算加权 R²。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, EvalPrediction, EarlyStoppingCallback
from accelerate import Accelerator
import timm
import math

TARGET_COLS = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")
AUX_COLS = ("Pre_GSHH_NDVI", "Height_Ave_cm")
WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}

# =============================================================================
# Fold helpers (ported from fold_split.ipynb)
# =============================================================================
SEED = 42
N_SPLITS = 5

def fix_zero_cells_by_move(
    meta: pd.DataFrame,
    key_col: str = "stratify_key",
    group_col: str = "image_id",
    fold_col: str = "fold",
    max_passes: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meta = meta.copy()
    for _ in range(max_passes):
        ct = pd.crosstab(meta[fold_col], meta[key_col])
        zeros = np.argwhere(ct.to_numpy() == 0)
        if len(zeros) == 0:
            break
        changed = False
        folds = ct.index.to_list()
        keys = ct.columns.to_list()
        rng.shuffle(zeros)
        for i, j in zeros:
            recv_fold = folds[i]
            key = keys[j]
            col = ct[key]
            donor_fold = col.idxmax()
            if col.loc[donor_fold] <= 1:
                continue
            candidates = meta[(meta[fold_col] == donor_fold) & (meta[key_col] == key)]
            if candidates.empty:
                continue
            pick = candidates.sample(1, random_state=int(rng.integers(0, 1_000_000_000))).index[0]
            meta.loc[pick, fold_col] = recv_fold
            ct = pd.crosstab(meta[fold_col], meta[key_col])
            changed = True
        if not changed:
            break
    return meta

def add_season_column(df: pd.DataFrame, date_col: str = "Sampling_Date") -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    m = df[date_col].dt.month
    df["Season"] = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8]), m.isin([9, 10, 11])],
        ["summer", "autumn", "winter", "spring"],
        default="unknown",
    )
    return df

def make_folds_state_season(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df = add_season_column(df, "Sampling_Date")

    meta = df.drop_duplicates("image_id")[["image_id", "State", "Season"]].copy().reset_index(drop=True)
    meta["State"] = meta["State"].fillna("Unknown")
    meta["Season"] = meta["Season"].fillna("Unknown")
    meta["stratify_key"] = meta["State"].astype(str) + "|" + meta["Season"].astype(str)

    meta["fold"] = -1
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        for fold, (_, va_idx) in enumerate(sgkf.split(meta, y=meta["stratify_key"], groups=meta["image_id"])):
            meta.loc[va_idx, "fold"] = fold
    except Exception:
        rng = np.random.default_rng(SEED)
        keys = meta["stratify_key"].unique().tolist()
        rng.shuffle(keys)
        fold_counts = [0] * N_SPLITS
        fold_key_counts = [dict() for _ in range(N_SPLITS)]
        global_key_counts = meta["stratify_key"].value_counts().to_dict()
        total_n = len(meta)
        key_groups = meta.groupby("stratify_key")["image_id"].apply(list).to_dict()
        keys_sorted = sorted(key_groups.keys(), key=lambda k: len(key_groups[k]), reverse=True)

        for k in keys_sorted:
            imgs = key_groups[k]
            rng.shuffle(imgs)
            for image_id in imgs:
                best_fold = None
                best_score = None
                for f in range(N_SPLITS):
                    new_total = fold_counts[f] + 1
                    desired_total = total_n / N_SPLITS
                    new_key_cnt = fold_key_counts[f].get(k, 0) + 1
                    desired_key = global_key_counts[k] / N_SPLITS
                    score = (new_total - desired_total) ** 2 + 3.0 * (new_key_cnt - desired_key) ** 2
                    if best_score is None or score < best_score:
                        best_score = score
                        best_fold = f
                fold_counts[best_fold] += 1
                fold_key_counts[best_fold][k] = fold_key_counts[best_fold].get(k, 0) + 1
                meta.loc[meta["image_id"] == image_id, "fold"] = best_fold
        undecided = meta["fold"] < 0
        if undecided.any():
            meta.loc[undecided, "fold"] = rng.integers(0, N_SPLITS, size=undecided.sum())

    df = df.merge(meta[["image_id", "fold"]], on="image_id", how="left")
    return df

def add_fold_column(df: pd.DataFrame) -> pd.DataFrame:
    df = make_folds_state_season(df)
    meta = df.drop_duplicates("image_id")[["image_id", "State", "Sampling_Date", "fold"]].copy()
    meta = add_season_column(meta, "Sampling_Date")
    meta["State"] = meta["State"].fillna("Unknown")
    meta["Season"] = meta["Season"].fillna("Unknown")
    meta["stratify_key"] = meta["State"].astype(str) + "|" + meta["Season"].astype(str)
    meta_fixed = fix_zero_cells_by_move(meta, key_col="stratify_key", group_col="image_id", fold_col="fold")
    df = df.drop(columns=["fold"]).merge(meta_fixed[["image_id", "fold"]], on="image_id", how="left")
    return df


# =============================================================================
# 模型
# =============================================================================
class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hid = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, depth: int = 2, patch: Tuple[int, int] = (2, 2), dropout: float = 0.0):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
        )
        self.patch = patch
        self.transformer = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
        )
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_feat = self.local(x)
        bsz, ch, h, w = local_feat.shape
        ph, pw = self.patch
        new_h = math.ceil(h / ph) * ph
        new_w = math.ceil(w / pw) * pw
        if new_h != h or new_w != w:
            local_feat = F.interpolate(local_feat, size=(new_h, new_w), mode="bilinear", align_corners=False)
            h, w = new_h, new_w

        tokens = local_feat.unfold(2, ph, ph).unfold(3, pw, pw)  # B,C,nh,nw,ph,pw
        tokens = tokens.contiguous().view(bsz, ch, -1, ph, pw)
        tokens = tokens.permute(0, 2, 3, 4, 1).reshape(bsz, -1, ch)

        for blk in self.transformer:
            tokens = blk(tokens)

        feat = tokens.view(bsz, -1, ph * pw, ch).permute(0, 3, 1, 2)
        nh = h // ph
        nw = w // pw
        feat = feat.view(bsz, ch, nh, nw, ph, pw).permute(0, 1, 2, 4, 3, 5)
        feat = feat.reshape(bsz, ch, h, w)

        if feat.shape[-2:] != x.shape[-2:]:
            feat = F.interpolate(feat, size=x.shape[-2:], mode="bilinear", align_corners=False)

        out = self.fuse(torch.cat([x, feat], dim=1))
        return out


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, sr_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        bsz, n, ch = x.shape
        q = self.q(x).reshape(bsz, n, self.heads, ch // self.heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            h, w = hw
            feat = x.transpose(1, 2).reshape(bsz, ch, h, w)
            feat = self.sr(feat)
            feat = feat.reshape(bsz, ch, -1).transpose(1, 2)
            feat = self.norm(feat)
        else:
            feat = x

        kv = self.kv(feat)
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(bsz, -1, self.heads, ch // self.heads).permute(0, 2, 3, 1)
        v = v.reshape(bsz, -1, self.heads, ch // self.heads).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(bsz, n, ch)
        out = self.proj(out)
        return out


class PVTBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, sr_ratio: int = 2, dropout: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.sra = SpatialReductionAttention(dim, heads=heads, sr_ratio=sr_ratio, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        x = x + self.sra(self.norm1(x), hw)
        x = x + self.ff(self.norm2(x))
        return x


class LocalMambaBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = (x * g).transpose(1, 2)
        x = self.dwconv(x).transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class T2TRetokenizer(nn.Module):
    def __init__(self, dim: int, depth: int = 2, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(depth)]
        )

    def forward(self, tokens: torch.Tensor, grid_hw: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _, ch = tokens.shape
        h, w = grid_hw
        feat_map = tokens.transpose(1, 2).reshape(bsz, ch, h, w)
        seq = feat_map.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq)
        seq_map = seq.transpose(1, 2).reshape(bsz, ch, h, w)
        pooled = F.adaptive_avg_pool2d(seq_map, (2, 2))
        retokens = pooled.flatten(2).transpose(1, 2)
        return retokens, seq_map


class CrossScaleFusion(nn.Module):
    def __init__(self, dim: int, heads: int = 6, dropout: float = 0.0, layers: int = 2):
        super().__init__()
        self.layers_s = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)]
        )
        self.layers_b = nn.ModuleList(
            [AttentionBlock(dim, heads=heads, dropout=dropout, mlp_ratio=2.0) for _ in range(layers)]
        )
        self.cross_s = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                for _ in range(layers)
            ]
        )
        self.cross_b = nn.ModuleList(
            [
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim)
                for _ in range(layers)
            ]
        )
        self.norm_s = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)

    def forward(self, tok_s: torch.Tensor, tok_b: torch.Tensor) -> torch.Tensor:
        bsz, _, ch = tok_s.shape
        cls_s = tok_s.new_zeros(bsz, 1, ch)
        cls_b = tok_b.new_zeros(bsz, 1, ch)
        tok_s = torch.cat([cls_s, tok_s], dim=1)
        tok_b = torch.cat([cls_b, tok_b], dim=1)

        for ls, lb, cs, cb in zip(self.layers_s, self.layers_b, self.cross_s, self.cross_b):
            tok_s = ls(tok_s)
            tok_b = lb(tok_b)
            q_s = self.norm_s(tok_s[:, :1])
            q_b = self.norm_b(tok_b[:, :1])
            cls_s_upd, _ = cs(q_s, torch.cat([tok_b, q_b], dim=1), torch.cat([tok_b, q_b], dim=1), need_weights=False)
            cls_b_upd, _ = cb(q_b, torch.cat([tok_s, q_s], dim=1), torch.cat([tok_s, q_s], dim=1), need_weights=False)
            tok_s = torch.cat([tok_s[:, :1] + cls_s_upd, tok_s[:, 1:]], dim=1)
            tok_b = torch.cat([tok_b[:, :1] + cls_b_upd, tok_b[:, 1:]], dim=1)

        tokens = torch.cat([tok_s[:, :1], tok_b[:, :1], tok_s[:, 1:], tok_b[:, 1:]], dim=1)
        return tokens


class PyramidMixer(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dims: Tuple[int, int, int],
        mobilevit_heads: int = 4,
        mobilevit_depth: int = 2,
        sra_heads: int = 6,
        sra_ratio: int = 2,
        mamba_depth: int = 3,
        mamba_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        c1, c2, c3 = dims
        self.proj1 = nn.Linear(dim_in, c1)
        self.mobilevit = MobileViTBlock(c1, heads=mobilevit_heads, depth=mobilevit_depth, dropout=dropout)
        self.proj2 = nn.Linear(c1, c2)
        self.pvt = PVTBlock(c2, heads=sra_heads, sr_ratio=sra_ratio, dropout=dropout, mlp_ratio=3.0)
        self.mamba_local = LocalMambaBlock(c2, kernel_size=mamba_kernel, dropout=dropout)
        self.proj3 = nn.Linear(c2, c3)
        self.mamba_global = nn.ModuleList(
            [LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)]
        )
        self.final_attn = AttentionBlock(c3, heads=min(8, c3 // 64 + 1), dropout=dropout, mlp_ratio=2.0)

    def _tokens_to_map(self, tokens: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        bsz, n, ch = tokens.shape
        h, w = target_hw
        need = h * w
        if n < need:
            pad = tokens.new_zeros(bsz, need - n, ch)
            tokens = torch.cat([tokens, pad], dim=1)
        tokens = tokens[:, :need, :]
        feat_map = tokens.transpose(1, 2).reshape(bsz, ch, h, w)
        return feat_map

    @staticmethod
    def _fit_hw(n_tokens: int) -> Tuple[int, int]:
        h = int(math.sqrt(n_tokens))
        w = h
        while h * w < n_tokens:
            w += 1
            if h * w < n_tokens:
                h += 1
        return h, w

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, n, _ = tokens.shape
        map_hw = (3, 4)
        t1 = self.proj1(tokens)
        m1 = self._tokens_to_map(t1, map_hw)
        m1 = self.mobilevit(m1)
        t1_out = m1.flatten(2).transpose(1, 2)[:, :n]

        t2 = self.proj2(t1_out)
        new_len = max(4, n // 2)
        t2 = t2[:, :new_len] + F.adaptive_avg_pool1d(t2.transpose(1, 2), new_len).transpose(1, 2)
        hw2 = self._fit_hw(t2.size(1))
        if t2.size(1) < hw2[0] * hw2[1]:
            pad = t2.new_zeros(bsz, hw2[0] * hw2[1] - t2.size(1), t2.size(2))
            t2 = torch.cat([t2, pad], dim=1)
        t2 = self.pvt(t2, hw2)
        t2 = self.mamba_local(t2)

        t3 = self.proj3(t2)
        pooled = torch.stack([t3.mean(dim=1), t3.max(dim=1).values], dim=1)
        t3 = pooled
        for blk in self.mamba_global:
            t3 = blk(t3)
        t3 = self.final_attn(t3)
        global_feat = t3.mean(dim=1)
        return global_feat, {"stage1_map": m1, "stage2_tokens": t2, "stage3_tokens": t3}


class CrossPVT_T2T_MambaHead(nn.Module):
    def __init__(self, cfg: HFConfig):
        super().__init__()
        self.feat_dim = cfg.vit_feat_dim
        self.token_embed = nn.Linear(cfg.vit_feat_dim, cfg.token_embed_dim)
        self.t2t = T2TRetokenizer(cfg.token_embed_dim, depth=cfg.t2t_depth, heads=cfg.cross_heads, dropout=cfg.dropout)
        self.cross = CrossScaleFusion(
            cfg.token_embed_dim, heads=cfg.cross_heads, dropout=cfg.dropout, layers=cfg.cross_layers
        )
        self.pyramid = PyramidMixer(
            dim_in=cfg.token_embed_dim,
            dims=cfg.pyramid_dims,
            mobilevit_heads=cfg.mobilevit_heads,
            mobilevit_depth=cfg.mobilevit_depth,
            sra_heads=cfg.sra_heads,
            sra_ratio=cfg.sra_ratio,
            mamba_depth=cfg.mamba_depth,
            mamba_kernel=cfg.mamba_kernel,
            dropout=cfg.dropout,
        )

        combined = cfg.pyramid_dims[-1]
        hidden = max(32, int(combined * cfg.hidden_ratio))

        def head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(combined, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden, 1),
            )

        self.head_green = head()
        self.head_clover = head()
        self.head_dead = head()
        self.score_head = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, 1))
        self.aux_head = (
            nn.Sequential(nn.LayerNorm(cfg.pyramid_dims[1]), nn.Linear(cfg.pyramid_dims[1], len(TARGET_COLS)))
            if cfg.aux_head
            else None
        )
        self.softplus = nn.Softplus(beta=1.0)

    def forward(
        self,
        tokens_small: torch.Tensor,
        tokens_big: torch.Tensor,
        small_grid: Tuple[int, int],
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        tokens_small = self.token_embed(tokens_small)
        tokens_big = self.token_embed(tokens_big)

        t2, stage1_map = self.t2t(tokens_small, small_grid)
        fused = self.cross(t2, tokens_big)
        feat, feat_maps = self.pyramid(fused)
        feat_maps["stage1_map"] = stage1_map

        green_pos = self.softplus(self.head_green(feat))
        clover_pos = self.softplus(self.head_clover(feat))
        dead_pos = self.softplus(self.head_dead(feat))


        out: Dict[str, torch.Tensor] = {
            "green": green_pos,
            "dead": dead_pos,
            "clover": clover_pos,
            "score_feat": feat,
        }
        if self.aux_head is not None:
            aux_tokens = feat_maps["stage2_tokens"]
            aux_pred = self.softplus(self.aux_head(aux_tokens.mean(dim=1)))
            out["aux"] = aux_pred
        if return_features:
            out["feature_maps"] = {
                "stage1": feat_maps.get("stage1_map"),
                "stage3": feat_maps.get("stage3_tokens"),
            }
        return out

# =============================================================================
# 配置
# =============================================================================
@dataclass
class HFConfig:
    train_csv: Path = Path("csiro-biomass/train.csv")
    image_dir: Path = Path("csiro-biomass")
    output_dir: Path = Path("runs/hf_trainer")
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    aux_head: bool = True
    label_smooth: float = 0.0
    bf16: bool = False

    small_grid: Tuple[int, int] = (2, 4)
    big_grid: Tuple[int, int] = (1, 2)
    pyramid_dims: Tuple[int, int, int] = (768, 1024, 1280)
    mamba_depth: int = 3
    mamba_kernel: int = 5
    mobilevit_heads: int = 4
    mobilevit_depth: int = 2
    sra_heads: int = 8
    sra_ratio: int = 2
    cross_heads: int = 8
    cross_layers: int = 2
    t2t_depth: int = 2

    vit_name: str = "vit_7b_patch16_dinov3.lvd1689m"
    vit_checkpoint: Path = Path("timm/vit_7b_patch16_dinov3.lvd1689m/model.safetensors")
    vit_input_size: int = 256
    vit_feat_dim: int = 4096
    token_embed_dim: int = 1024
    
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Feature extractor (ViT) separated from head
# =============================================================================
def build_feature_extractor(cfg: HFConfig) -> tuple[nn.Module, callable]:
    if not cfg.vit_checkpoint.exists():
        raise FileNotFoundError(f"ViT checkpoint not found: {cfg.vit_checkpoint}")
    base = timm.create_model(
        cfg.vit_name,
        pretrained=False,
        num_classes=0,
        global_pool="avg",
        checkpoint_path=str(cfg.vit_checkpoint),
    )
    for p in base.parameters():
        p.requires_grad = False
    base.eval()
    if not torch.cuda.is_available():
        print(">>> 未检测到 GPU，ViT 特征提取将在 CPU 上运行（较慢）")
    base = base.to(cfg.device).to(torch.bfloat16)
    data_config = timm.data.resolve_model_data_config(base)
    timm_transform = timm.data.create_transform(**data_config, is_training=False)
    return base, timm_transform

@torch.no_grad()
def encode_tiles(
    images: torch.Tensor,
    backbone: nn.Module,
    backbone_transform,
    grid: Tuple[int, int],
) -> torch.Tensor:
    bsz, ch, h, w = images.shape
    r, c = grid
    hs = torch.linspace(0, h, steps=r + 1, device=images.device).round().long()
    ws = torch.linspace(0, w, steps=c + 1, device=images.device).round().long()
    tiles: List[torch.Tensor] = []
    for i in range(r):
        for j in range(c):
            rs, re = hs[i].item(), hs[i + 1].item()
            cs, ce = ws[j].item(), ws[j + 1].item()
            xt = images[:, :, rs:re, cs:ce]
            tiles.append(xt)
    tiles = torch.stack(tiles, dim=1)
    flat = tiles.view(-1, ch, tiles.shape[-2], tiles.shape[-1])

    # timm 变换（包含 resize + normalize），逐 tile 处理；使用 CPU 以保证兼容性，再移回设备
    proc = []
    for t in flat:
        proc.append(backbone_transform(t.to(torch.float16)).unsqueeze(0))
    proc = torch.cat(proc, dim=0).to(images.device).to(torch.bfloat16)

    assert proc.max() < 6 and proc.min() > -6, f"proc is not normalized to [0, 1] {proc.max()}"

    feats = backbone(proc)
    feats = feats.view(bsz, -1, feats.shape[-1])
    return feats

def pack_targets(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    clover = out["gdm"] - out["green"]
    dead = out["total"] - out["gdm"]
    return torch.cat([out["green"], dead, clover, out["gdm"], out["total"]], dim=1)

# =============================================================================
# 数据集与增强
# =============================================================================
def build_full_image_transform(train: bool) -> A.Compose:
    aug = [
        A.OneOf(
            [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
            ],
            p=0.8 if train else 0.0,
        ),
        A.RandomBrightnessContrast(0.15, 0.15, p=0.6 if train else 0.0),
        A.HueSaturationValue(8, 12, 8, p=0.4 if train else 0.0),
        A.RandomGamma(gamma_limit=(85, 115), p=0.25 if train else 0.0),
        A.OneOf(
            [
                A.ImageCompression(quality_range=(55, 95)),
                A.GaussNoise(std_range=(0.02, 0.10)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=3),
            ],
            p=0.30 if train else 0.0,
        ),
        A.Normalize(mean=(0,0,0), std=(1,1,1), max_pixel_value=255.0),  # 先缩放到 0-1
        ToTensorV2(),
    ]
    return A.Compose(aug)


class BiomassHFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, train: bool):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = build_full_image_transform(train=train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / row["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(image=img)["image"]
        # 提供完整 5 列标签，顺序与 TARGET_COLS 对齐
        labels = torch.tensor([row[col] for col in TARGET_COLS], dtype=torch.float32)
        return {"pixel_values": img_t, "labels": labels}


def data_collator(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


# =============================================================================
# 模型包装给 Trainer
# =============================================================================
class HFWrapper(nn.Module):
    def __init__(self, cfg: HFConfig, backbone: nn.Module, backbone_transform):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.backbone_transform = backbone_transform
        self.head = CrossPVT_T2T_MambaHead(
            HFConfig(
                dropout=cfg.dropout,
                hidden_ratio=cfg.hidden_ratio,
                aux_head=cfg.aux_head,
                small_grid=cfg.small_grid,
                big_grid=cfg.big_grid,
                pyramid_dims=cfg.pyramid_dims,
                mamba_depth=cfg.mamba_depth,
                mamba_kernel=cfg.mamba_kernel,
                mobilevit_heads=cfg.mobilevit_heads,
                mobilevit_depth=cfg.mobilevit_depth,
                sra_heads=cfg.sra_heads,
                sra_ratio=cfg.sra_ratio,
                cross_heads=cfg.cross_heads,
                cross_layers=cfg.cross_layers,
                t2t_depth=cfg.t2t_depth,
                vit_name=cfg.vit_name,
                vit_checkpoint=cfg.vit_checkpoint,
                vit_feat_dim=4096,
                token_embed_dim=1024,
            )
        )
        self.loss_aux_w = 0.4
        self.label_smooth = cfg.label_smooth

    def forward(self, pixel_values=None, labels=None):
        tok_small = encode_tiles(pixel_values, self.backbone, self.backbone_transform, self.cfg.small_grid)
        tok_big = encode_tiles(pixel_values, self.backbone, self.backbone_transform, self.cfg.big_grid)
        out = self.head(tok_small, tok_big, self.cfg.small_grid, return_features=False)

        green = out["green"]
        clover = out["clover"]
        dead = out["dead"]

        gdm = green + clover
        total = green + clover + dead

        logits_loss = torch.cat([green, dead, clover], dim=1)
        logits_result = torch.cat([green, dead, clover, gdm, total], dim=1)
        result = {"logits": logits_result}
        if labels is not None:
            loss_main = F.smooth_l1_loss(logits_loss, labels[:,:3], beta=self.label_smooth)
            loss_aux = 0.0
            if self.cfg.aux_head and "aux" in out:
                loss_aux = F.smooth_l1_loss(out["aux"], labels)
            loss = loss_main + self.loss_aux_w * loss_aux
            result["loss"] = loss
        return result

    def state_dict(self, *args, **kwargs):
        # 保存时排除 backbone 权重，避免巨大的 7B 体积
        sd = super().state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if not k.startswith("backbone.")}

    def load_state_dict(self, state_dict, strict: bool = False):
        # 加载时忽略缺失的 backbone 键，确保兼容过滤后的 state_dict
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("backbone.")}
        return super().load_state_dict(filtered, strict=False)


# =============================================================================
# 评估指标（加权 R²）
# =============================================================================
def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:

    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = torch.as_tensor(preds, dtype=torch.float32)
    y_true = torch.as_tensor(eval_pred.label_ids, dtype=torch.float32)

    w_base = torch.tensor([WEIGHTS[c] for c in TARGET_COLS], dtype=torch.float32)
    w = w_base.unsqueeze(0).expand(y_true.size(0), -1)
    w_flat = w.reshape(-1)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    y_w = (w_flat * y_true_flat).sum() / w_flat.sum()
    ss_res = (w_flat * (y_true_flat - y_pred_flat) ** 2).sum()
    ss_tot = (w_flat * (y_true_flat - y_w) ** 2).sum()
    r2 = 1.0 - (ss_res / ss_tot)

    total_res = 0.0
    summary: List[tuple[str, float, float]] = []
    for idx, name in enumerate(TARGET_COLS):
        wt = WEIGHTS[name]
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        res = (wt * (y_t - y_p) ** 2).sum()
        tot = (wt * (y_t - y_w) ** 2).sum()
        total_res += float(res)
        summary.append((name, float(res), float(tot)))

    print(f"  weighted R2 = {float(r2):.5f}")
    tot_sum = sum(s[2] for s in summary)
    if total_res > 0 and tot_sum > 0:
        for name, res, tot in sorted(summary, key=lambda x: x[1], reverse=True):
            res_share = res / total_res
            tot_share = tot / tot_sum
            print(f"    {name}: res_share={res_share:.3f}, tot_share={tot_share:.3f}")

    return {"weighted_r2": float(r2)}


# =============================================================================
# 主流程
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Train with HF Trainer")
    ap.add_argument("--train-csv", type=Path, default=HFConfig.train_csv)
    ap.add_argument("--image-dir", type=Path, default=HFConfig.image_dir)
    ap.add_argument("--output-dir", type=Path, default=HFConfig.output_dir)
    ap.add_argument("--folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=HFConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=HFConfig.batch_size)
    ap.add_argument("--num-workers", type=int, default=HFConfig.num_workers)
    ap.add_argument("--lr", type=float, default=HFConfig.lr)
    ap.add_argument("--weight-decay", type=float, default=HFConfig.weight_decay)
    ap.add_argument("--dropout", type=float, default=HFConfig.dropout)
    ap.add_argument("--hidden-ratio", type=float, default=HFConfig.hidden_ratio)
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    folds = tuple(int(x) for x in str(args.folds).split(",") if x.strip() != "")
    cfg = HFConfig(
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        folds=folds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_ratio=args.hidden_ratio,
        aux_head=not args.no_aux,
        bf16=args.bf16,
    )

    # 数据划分
    df = pd.read_csv(cfg.train_csv)
    df = add_fold_column(df)
    wide = (
        df.pivot_table(index=["image_id", "image_path", "fold"], columns="target_name", values="target")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    # Backbone + timm transform
    backbone, backbone_transform = build_feature_extractor(HFConfig(device=device))

    for fold in folds:
        print(f"\n===== Fold {fold} / {len(folds)} =====")
        train_df = wide[wide["fold"] != fold].reset_index(drop=True)
        val_df = wide[wide["fold"] == fold].reset_index(drop=True)

        train_ds = BiomassHFDataset(train_df, cfg.image_dir, train=True)
        val_ds = BiomassHFDataset(val_df, cfg.image_dir, train=False)

        model = HFWrapper(cfg, backbone, backbone_transform)

        args_tr = TrainingArguments(
            output_dir=str(cfg.output_dir / f"fold{fold}"),
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            learning_rate=cfg.lr,
            weight_decay=cfg.weight_decay,
            dataloader_num_workers=cfg.num_workers,
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            bf16=cfg.bf16,
            fp16=not cfg.bf16,
            save_total_limit=10,
            load_best_model_at_end=True,
            metric_for_best_model="weighted_r2",
            greater_is_better=True,
            report_to="tensorboard",
        )

        trainer = Trainer(
            model=model,
            args=args_tr,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
        )

        trainer.train()
        metrics = trainer.evaluate()
        print(f"Fold {fold} eval metrics: {metrics}")


if __name__ == "__main__":
    main()
