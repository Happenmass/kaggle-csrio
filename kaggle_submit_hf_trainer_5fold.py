# %%
"""
Kaggle 提交示例（Trainer + Dataset，5-fold head ensemble，timm ViT 单独加载）

你需要做的只有两件事：
1) 把本文件顶部的全局路径变量改成你 Kaggle 实际的 /kaggle/input/... 目录
2) 确保 5 个 fold 的 best head 权重已按 A 格式存在：
   {HEAD_5FOLD_DIR}/fold0/model.safetensors
   ...
   {HEAD_5FOLD_DIR}/fold4/model.safetensors

说明：
- 这些 head safetensors 来自你训练脚本 train_hf_trainer.py（HF Trainer 保存的 model.safetensors）
- 权重不包含 timm ViT，所以推理时会单独加载 ViT，并把 ViT 特征喂给 head
- 推理使用 Trainer.predict(...)，并对 5 折输出做 simple mean
"""

# %%
import os
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from safetensors.torch import load_file as safetensors_load_file
from transformers import Trainer, TrainingArguments

# %%
# =============================================================================
# 需要你按 Kaggle 实际情况替换的全局路径变量（我先假设一套）
# =============================================================================
KAGGLE_DATA_DIR = Path("/home/ecs-user/code/happen/kaggle-csrio")  # 里面有 csiro-biomass/test.csv + test/...
KAGGLE_VIT_DIR = Path("/home/ecs-user/code/happen/kaggle-csrio/timm/vit_7b_patch16_dinov3.lvd1689m")  # 可选：本地 vit 权重目录（里面有 model.safetensors）
KAGGLE_HEAD_5FOLD_DIR = Path("/home/ecs-user/code/happen/kaggle-csrio/runs/hf_trainer_A")  # A 格式目录

# 输出目录（Kaggle 工作目录可写）
OUTPUT_DIR = Path("./kaggle_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 你训练时的相对数据目录结构（与仓库一致）
TEST_CSV = KAGGLE_DATA_DIR / "csiro-biomass" / "test.csv"
IMAGE_ROOT = KAGGLE_DATA_DIR / "csiro-biomass"
SAMPLE_SUB = KAGGLE_DATA_DIR / "csiro-biomass" / "sample_submission.csv"

# 5-fold
FOLDS = (0, 1, 2, 3, 4)

# 单卡建议：batch 尽量小（ViT-7B 很吃显存）
PER_DEVICE_BATCH_SIZE = 1
NUM_WORKERS = 1

# =============================================================================
# Fast 推理（不使用 Trainer）：两阶段（ViT 预计算缓存 -> head 推理），双进程/双卡并行
# =============================================================================
FAST_INFER = True
GPU_IDS = (0, 1)  # 物理 GPU id（Kaggle 双卡常见是 0/1）
VIT_BATCH_SIZE = 2  # ViT 预计算阶段 batch（太大容易 OOM）
HEAD_BATCH_SIZE = 32  # head 推理阶段 batch（通常可以大很多）
CACHE_DIR = OUTPUT_DIR / "vit_cache"
# ViT 预处理模式：
# - "exact_timm": 恢复原来的 backbone_transform 路径（timm.create_transform），最接近原版结果，但更慢
# - "fast_gpu": 纯 torch GPU resize+normalize（更快，但会产生可见差异）
VIT_PREPROCESS_MODE = "exact_timm"  # "exact_timm" or "fast_gpu"
# 缓存精度：
# - "fp16": 更快/更省磁盘，但会引入量化误差（submission 可能与非缓存版略有差异）
# - "fp32": 更接近“直接推理不落盘”的结果，但更占磁盘/IO
CACHE_DTYPE = "fp32"  # "fp16" or "fp32"

# Head 推理精度策略（为了更贴近 Trainer 的 AMP 行为）
# - head 权重保持 fp32，forward 用 autocast（很多算子会自动选择更稳定的内部精度）
HEAD_WEIGHT_DTYPE = "fp32"  # "fp32"（推荐）或 "fp16"/"bf16"
USE_AUTOCAST_FOR_HEAD = True

# ViT 推理时是否也启用 autocast（进一步贴近 Trainer/AMP 行为）
USE_AUTOCAST_FOR_VIT = True

# Notebook 里 multiprocessing(spawn) 经常会因为函数无法从 __main__ 导入而报：
# "Can't get attribute '_vit_precompute_worker' on <module '__main__' ...>"
# 为了在 notebook 里也能双卡并行，这里提供线程后端（同一进程两个线程分别绑定两张 GPU）。
# - "process": 用多进程（更隔离，Kaggle Script 场景推荐）
# - "thread": 用多线程（Notebook 场景推荐，避免 spawn pickle/import 问题）
PARALLEL_BACKEND = "thread"  # "thread" or "process"
VIT_INIT_CPU_ONCE_CLONE_TO_GPU = True  # 仅 thread 后端生效：CPU 只加载 1 次 ViT，然后复制到两张 GPU


def _in_notebook() -> bool:
    return "ipykernel" in sys.modules or "IPython" in sys.modules


def _run_parallel_jobs(jobs: list[tuple[callable, dict]], *, backend: str) -> None:
    """
    jobs: [(fn, kwargs), ...]
    backend:
      - "process": spawn 子进程执行（要求 fn 可 picklable 且在可 import 的模块顶层定义）
      - "thread": 线程执行（Notebook 友好；同进程共享内存，需注意不要写同一段 memmap）
    """
    if backend == "process":
        ctx = mp.get_context("spawn")
        procs = []
        for fn, kwargs in jobs:
            p = ctx.Process(target=fn, kwargs=kwargs)
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"子进程异常退出：exitcode={p.exitcode}")
        return

    # thread backend
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(fn, **kwargs) for fn, kwargs in jobs]
        for fut in as_completed(futs):
            # re-raise any exception
            fut.result()


# thread 后端下的共享 ViT（每张 GPU 一份），避免每个线程各自从磁盘/Hub 加载
_VIT_THREAD_MODELS: dict[int, tuple[nn.Module, object]] = {}


def _create_vit_model_uninitialized_on_device(device: torch.device) -> nn.Module:
    """
    在指定 device 上创建与 ViT-7B 同结构的模型，但不从磁盘/Hub 加载权重（pretrained=False）。
    通过 torch.set_default_device 尽量避免在 CPU 上分配巨型参数（对 7B 很关键）。
    """
    # torch.set_default_device 是全局的，尽量短时间使用
    torch.set_default_device(str(device))
    try:
        if CFG.vit_load_from_hf_hub:
            m = timm.create_model(
                CFG.vit_hf_hub_id,
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
        else:
            m = timm.create_model(
                CFG.vit_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
    finally:
        torch.set_default_device("cpu")
    return m


def _init_vit_models_for_thread_backend(gpu_ids: Tuple[int, int], dtype: torch.dtype) -> None:
    """
    CPU 上加载一次预训练 ViT，然后复制到多个 GPU（每个 GPU 一份），供 thread 后端并行使用。
    """
    global _VIT_THREAD_MODELS
    if _VIT_THREAD_MODELS:
        return

    # 1) CPU 加载一次（占用一次 CPU 内存）
    cpu_device = torch.device("cpu")
    if CFG.vit_load_from_hf_hub:
        cpu_model = timm.create_model(
            CFG.vit_hf_hub_id,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        ).to(cpu_device)
    else:
        if not CFG.vit_checkpoint.exists():
            raise FileNotFoundError(f"ViT checkpoint 不存在: {CFG.vit_checkpoint}")
        cpu_model = timm.create_model(
            CFG.vit_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            checkpoint_path=str(CFG.vit_checkpoint),
        ).to(cpu_device)

    cpu_model.eval()
    data_config = timm.data.resolve_model_data_config(cpu_model)
    backbone_transform = timm.data.create_transform(**data_config, is_training=False)
    state = cpu_model.state_dict()  # 引用 CPU 权重张量（不会额外复制）

    # 2) 逐 GPU 创建模型并加载权重（CPU -> GPU copy）
    for dev_id in gpu_ids:
        torch.cuda.set_device(dev_id)
        dev = torch.device(f"cuda:{dev_id}")
        m = _create_vit_model_uninitialized_on_device(dev)
        missing, unexpected = m.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"ViT load_state_dict 不匹配：missing={missing[:5]} unexpected={unexpected[:5]}")
        m.eval()
        m = m.to(device=dev, dtype=dtype)
        _VIT_THREAD_MODELS[dev_id] = (m, backbone_transform)

    # 3) 释放 CPU 模型与 state（此时 GPU 已各自持有权重）
    del cpu_model
    del state
    gc.collect()


# -----------------------------------------------------------------------------
# 兼容旧的 Trainer 推理路径（如你需要回退）
# -----------------------------------------------------------------------------
USE_MODEL_PARALLEL = True
BACKBONE_DEVICE_STR = "cuda:0"
HEAD_DEVICE_STR = "cuda:1"

# 是否做推理 TTA（会更慢；你当前要求是“平均”，默认不开）
USE_TTA = False

# %%
# =============================================================================
# 目标列定义（需与训练一致）
# =============================================================================
TARGET_COLS = ("Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g")


# %%
# =============================================================================
# 训练/推理配置（需要与 train_hf_trainer.py 的 HFConfig 对齐）
# -----------------------------------------------------------------------------
# 如果你训练时改过 small_grid/big_grid/pyramid_dims/token_embed_dim 等结构超参，
# 请在这里同步修改，否则 load_state_dict 会报错（结构不匹配）。
# =============================================================================
class InferCFG:
    # ViT 加载方式：
    # - 推荐：Kaggle 开网时用 hf_hub 直接拉取预训练权重（你给的方式）
    # - 备选：不开网/想固定版本时，用本地 checkpoint_path
    vit_load_from_hf_hub: bool = False
    vit_hf_hub_id: str = "hf_hub:timm/vit_7b_patch16_dinov3.lvd1689m"
    vit_name: str = "vit_7b_patch16_dinov3.lvd1689m"  # 本地 ckpt 方式需要
    vit_checkpoint: Path = KAGGLE_VIT_DIR / "model.safetensors"  # 本地 ckpt 方式需要
    vit_feat_dim: int = 4096
    token_embed_dim: int = 1024

    # tile grid（与你 train_hf_trainer.py 一致）
    small_grid: Tuple[int, int] = (2, 4)
    big_grid: Tuple[int, int] = (1, 2)

    # head 结构（与你 train_hf_trainer.py 一致）
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    aux_head: bool = True
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


CFG = InferCFG()


# %%
# =============================================================================
# Head 结构（为保证脚本独立运行，这里直接内置一份与 train_hf_trainer.py 对齐的实现）
# =============================================================================
@dataclass
class HFConfig:
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    aux_head: bool = True

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
    vit_checkpoint: Path = Path("unused")
    vit_feat_dim: int = 4096
    token_embed_dim: int = 1024


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
    def __init__(
        self, dim: int, heads: int = 4, depth: int = 2, patch: Tuple[int, int] = (2, 2), dropout: float = 0.0
    ):
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
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim) for _ in range(layers)]
        )
        self.cross_b = nn.ModuleList(
            [nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True, kdim=dim, vdim=dim) for _ in range(layers)]
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
        self.mamba_global = nn.ModuleList([LocalMambaBlock(c3, kernel_size=mamba_kernel, dropout=dropout) for _ in range(mamba_depth)])
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
        self.cross = CrossScaleFusion(cfg.token_embed_dim, heads=cfg.cross_heads, dropout=cfg.dropout, layers=cfg.cross_layers)
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
        # 训练脚本里存在但未必参与 loss 的 score_head：为了 load_state_dict 完全一致，这里保留
        self.score_head = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, 1))
        self.aux_head = (
            nn.Sequential(nn.LayerNorm(cfg.pyramid_dims[1]), nn.Linear(cfg.pyramid_dims[1], len(TARGET_COLS))) if cfg.aux_head else None
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

        out: Dict[str, torch.Tensor] = {"green": green_pos, "dead": dead_pos, "clover": clover_pos, "score_feat": feat}
        if self.aux_head is not None:
            aux_tokens = feat_maps["stage2_tokens"]
            aux_pred = self.softplus(self.aux_head(aux_tokens.mean(dim=1)))
            out["aux"] = aux_pred
        if return_features:
            out["feature_maps"] = {"stage1": feat_maps.get("stage1_map"), "stage3": feat_maps.get("stage3_tokens")}
        return out


# %%
# =============================================================================
# 数据增强：推理用 train=False（只做 Normalize + ToTensor）
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
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ]
    return A.Compose(aug)


# %%
# =============================================================================
# Test Dataset：按“图片级别”去重，避免同一张图算 5 次 ViT
# =============================================================================
class BiomassHFTestImageDataset(Dataset):
    def __init__(self, test_csv: Path, image_root: Path):
        self.test_df = pd.read_csv(test_csv)
        # image_id = sample_id 的前缀（__ 之前）
        self.test_df["image_id"] = self.test_df["sample_id"].astype(str).str.split("__", n=1).str[0]
        img_df = self.test_df.drop_duplicates("image_id")[["image_id", "image_path"]].reset_index(drop=True)
        self.image_ids: List[str] = img_df["image_id"].tolist()
        self.image_paths: List[str] = img_df["image_path"].tolist()
        self.image_root = image_root
        self.transform = build_full_image_transform(train=False)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel = self.image_paths[idx]
        img_path = self.image_root / rel
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(image=img)["image"]
        return {"pixel_values": img_t}


def data_collator_infer(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    return {"pixel_values": pixel_values}


# %%
# =============================================================================
# timm ViT：单独加载 + 冻结
# =============================================================================
def pick_infer_dtype() -> torch.dtype:
    # Kaggle 常见 T4: bf16 不支持，优先 fp16；没有 cuda 就 fp32
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_vit_backbone(
    *,
    vit_name: str,
    vit_ckpt: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    # 优先：从 HuggingFace Hub 直接拉取 timm 模型（需要 Kaggle 开网）
    if CFG.vit_load_from_hf_hub:
        model = timm.create_model(
            CFG.vit_hf_hub_id,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
    else:
        if not vit_ckpt.exists():
            raise FileNotFoundError(f"ViT checkpoint 不存在: {vit_ckpt}")
        model = timm.create_model(
            vit_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            checkpoint_path=str(vit_ckpt),
        )
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model = model.to(device=device, dtype=dtype)

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)
    return model, transform


def _vit_mean_std() -> tuple[torch.Tensor, torch.Tensor]:
    # 来自 timm/vit_7b_patch16_dinov3.lvd1689m/config.json
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return mean, std


def _resize_norm_batch(x: torch.Tensor, *, out_hw: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    """
    x: [B,3,H,W]，值域假设在 [0,1]
    输出：resize 到 out_hw + timm mean/std normalize
    """
    # timm/torchvision 的 Resize 通常会做 antialias；这里打开 antialias 以尽量贴近 timm 行为
    x = F.interpolate(x, size=out_hw, mode="bicubic", align_corners=False, antialias=True)
    mean, std = _vit_mean_std()
    mean = mean.to(device=x.device, dtype=x.dtype)
    std = std.to(device=x.device, dtype=x.dtype)
    x = (x - mean) / std
    return x.to(dtype)


@torch.no_grad()
def encode_tiles_fast(
    images: torch.Tensor,
    backbone: nn.Module,
    grid: Tuple[int, int],
    *,
    dtype: torch.dtype,
    vit_input_hw: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    更快的 tile 编码：
    - 不走 timm.create_transform（避免逐 tile python 循环）
    - 直接在 GPU 上 resize + normalize
    """
    bsz, ch, h, w = images.shape
    r, c = grid
    hs = torch.linspace(0, h, steps=r + 1, device=images.device).round().long()
    ws = torch.linspace(0, w, steps=c + 1, device=images.device).round().long()

    tiles: List[torch.Tensor] = []
    for i in range(r):
        for j in range(c):
            rs, re = hs[i].item(), hs[i + 1].item()
            cs, ce = ws[j].item(), ws[j + 1].item()
            tiles.append(images[:, :, rs:re, cs:ce])
    tiles = torch.stack(tiles, dim=1)  # [B,T,C,Ht,Wt]
    flat = tiles.reshape(-1, ch, tiles.shape[-2], tiles.shape[-1])  # [B*T,C,Ht,Wt]
    flat = _resize_norm_batch(flat, out_hw=vit_input_hw, dtype=dtype)
    if USE_AUTOCAST_FOR_VIT and flat.is_cuda:
        with torch.autocast(device_type="cuda", dtype=dtype):
            feats = backbone(flat)  # [B*T, feat]
    else:
        feats = backbone(flat)  # [B*T, feat]
    feats = feats.view(bsz, -1, feats.shape[-1])  # [B, T, feat]
    return feats


@torch.no_grad()
def encode_tiles_exact_timm_transform(
    images: torch.Tensor,
    backbone: nn.Module,
    backbone_transform,
    grid: Tuple[int, int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    恢复原版路径：逐 tile 调用 timm.create_transform（backbone_transform），以最大程度复现原先输出。
    注意：该路径会比 fast_gpu 慢（存在 python for 循环）。
    """
    bsz, ch, h, w = images.shape
    r, c = grid
    hs = torch.linspace(0, h, steps=r + 1).round().long()
    ws = torch.linspace(0, w, steps=c + 1).round().long()

    tiles: List[torch.Tensor] = []
    for i in range(r):
        for j in range(c):
            rs, re = hs[i].item(), hs[i + 1].item()
            cs, ce = ws[j].item(), ws[j + 1].item()
            tiles.append(images[:, :, rs:re, cs:ce])
    tiles = torch.stack(tiles, dim=1)  # [B,T,C,Ht,Wt]
    flat = tiles.view(-1, ch, tiles.shape[-2], tiles.shape[-1])  # [B*T,C,Ht,Wt]

    # 与原始 train_hf_trainer.py 的做法一致：逐 tile 走 backbone_transform（通常在 CPU）
    proc_list = []
    for t in flat:
        proc_list.append(backbone_transform(t.to(torch.float16)).unsqueeze(0))
    proc = torch.cat(proc_list, dim=0).to(device=next(backbone.parameters()).device).to(dtype)

    if USE_AUTOCAST_FOR_VIT and proc.is_cuda:
        with torch.autocast(device_type="cuda", dtype=dtype):
            feats = backbone(proc)  # [B*T, feat]
    else:
        feats = backbone(proc)  # [B*T, feat]
    feats = feats.view(bsz, -1, feats.shape[-1])
    return feats


# 兼容旧实现签名：根据 VIT_PREPROCESS_MODE 选择 exact_timm 或 fast_gpu
@torch.no_grad()
def encode_tiles_infer(
    images: torch.Tensor,
    backbone: nn.Module,
    backbone_transform,
    grid: Tuple[int, int],
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if VIT_PREPROCESS_MODE == "exact_timm":
        return encode_tiles_exact_timm_transform(images, backbone, backbone_transform, grid, dtype=dtype)
    return encode_tiles_fast(images, backbone, grid, dtype=dtype)


# %%
# =============================================================================
# 5-fold ensemble 模型（一次 ViT 编码，多 head 平均）
# =============================================================================
class HFEnsembleWrapper(nn.Module):
    def __init__(
        self,
        *,
        cfg: InferCFG,
        backbone: nn.Module,
        backbone_transform,
        dtype: torch.dtype,
        backbone_device: torch.device,
        head_device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.backbone_device = backbone_device
        self.head_device = head_device

        # 关键：不要把 backbone 注册为 nn.Module 子模块（否则 Trainer 会把它一起 .to(head_device)，破坏模型并行）
        # 这里用 __dict__ 绕开 nn.Module.__setattr__ 的注册逻辑。
        self.__dict__["backbone"] = backbone
        self.__dict__["backbone_transform"] = backbone_transform

        # 用 HFConfig 创建 head（保持 train_hf_trainer.py 一致）
        head_cfg = HFConfig(
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
            vit_checkpoint=Path("unused_in_infer"),
            vit_feat_dim=cfg.vit_feat_dim,
            token_embed_dim=cfg.token_embed_dim,
        )

        self.heads = nn.ModuleList([CrossPVT_T2T_MambaHead(head_cfg) for _ in range(len(FOLDS))])

    def load_fold_head_from_safetensors(self, fold_idx: int, safetensors_path: Path) -> None:
        sd = safetensors_load_file(str(safetensors_path))
        # 训练时保存的是 HFWrapper 的 state_dict（不含 backbone），head 权重在 "head.*"
        head_sd = {k[len("head.") :]: v for k, v in sd.items() if k.startswith("head.")}
        missing, unexpected = self.heads[fold_idx].load_state_dict(head_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"fold{fold_idx} head load_state_dict 不匹配：missing={missing[:10]} unexpected={unexpected[:10]}"
            )

    def forward(self, pixel_values=None, labels=None):
        # Trainer.predict 会传 pixel_values；labels 为 None
        # pixel_values 默认会被 Trainer 放到“模型设备”（也就是 head_device），这里我们手动搬到 backbone_device 做 ViT 编码
        if pixel_values.device != self.backbone_device:
            pixel_values_bb = pixel_values.to(device=self.backbone_device, non_blocking=True)
        else:
            pixel_values_bb = pixel_values

        tok_small = encode_tiles_infer(
            pixel_values_bb, self.backbone, self.backbone_transform, self.cfg.small_grid, dtype=self.dtype
        )
        tok_big = encode_tiles_infer(
            pixel_values_bb, self.backbone, self.backbone_transform, self.cfg.big_grid, dtype=self.dtype
        )

        # 把 token 搬到 head_device 给 head 推理
        if tok_small.device != self.head_device:
            tok_small = tok_small.to(device=self.head_device, non_blocking=True)
        if tok_big.device != self.head_device:
            tok_big = tok_big.to(device=self.head_device, non_blocking=True)

        logits_list = []
        for head in self.heads:
            out = head(tok_small, tok_big, self.cfg.small_grid, return_features=False)
            green = out["green"]
            clover = out["clover"]
            dead = out["dead"]
            gdm = green + clover
            total = green + clover + dead
            logits = torch.cat([green, dead, clover, gdm, total], dim=1)  # [B, 5]
            logits_list.append(logits)

        logits_mean = torch.stack(logits_list, dim=0).mean(dim=0)
        return {"logits": logits_mean}


# %%
# =============================================================================
# 单 fold 模型（常驻 1 个 head；每个 fold 推理完后释放，再加载下一个 fold）
# =============================================================================
class HFSingleFoldWrapper(nn.Module):
    def __init__(
        self,
        *,
        cfg: InferCFG,
        backbone: nn.Module,
        backbone_transform,
        dtype: torch.dtype,
        backbone_device: torch.device,
        head_device: torch.device,
    ):
        super().__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.backbone_device = backbone_device
        self.head_device = head_device

        # 关键：告诉 HuggingFace Trainer 这是“模型并行”，避免：
        # - 自动 nn.DataParallel（多 GPU 时默认会包一层，破坏我们的 device 搬运逻辑）
        # - 自动把整个 model 迁移到 args.device（通常是 cuda:0）
        #
        # 参考：HF 内置的 model-parallel 模型也用这些标记来让 Trainer 跳过 DP 包装。
        self.is_parallelizable = True
        self.model_parallel = True
        self.device_map = {"backbone": str(backbone_device), "head": str(head_device)}

        # 关键：不要把 backbone 注册为 nn.Module 子模块（否则 Trainer 会把它一起 .to(head_device)，破坏模型并行）
        self.__dict__["backbone"] = backbone
        self.__dict__["backbone_transform"] = backbone_transform

        head_cfg = HFConfig(
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
            vit_checkpoint=Path("unused_in_infer"),
            vit_feat_dim=cfg.vit_feat_dim,
            token_embed_dim=cfg.token_embed_dim,
        )
        self.head = CrossPVT_T2T_MambaHead(head_cfg)

    def load_head_from_safetensors(self, safetensors_path: Path) -> None:
        sd = safetensors_load_file(str(safetensors_path))
        head_sd = {k[len("head.") :]: v for k, v in sd.items() if k.startswith("head.")}
        missing, unexpected = self.head.load_state_dict(head_sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"head load_state_dict 不匹配：missing={missing[:10]} unexpected={unexpected[:10]}")

    def forward(self, pixel_values=None, labels=None):
        if pixel_values.device != self.backbone_device:
            pixel_values_bb = pixel_values.to(device=self.backbone_device, non_blocking=True)
        else:
            pixel_values_bb = pixel_values

        tok_small = encode_tiles_infer(
            pixel_values_bb, self.backbone, self.backbone_transform, self.cfg.small_grid, dtype=self.dtype
        )
        tok_big = encode_tiles_infer(
            pixel_values_bb, self.backbone, self.backbone_transform, self.cfg.big_grid, dtype=self.dtype
        )

        if tok_small.device != self.head_device:
            tok_small = tok_small.to(device=self.head_device, non_blocking=True)
        if tok_big.device != self.head_device:
            tok_big = tok_big.to(device=self.head_device, non_blocking=True)

        out = self.head(tok_small, tok_big, self.cfg.small_grid, return_features=False)
        green = out["green"]
        clover = out["clover"]
        dead = out["dead"]
        gdm = green + clover
        total = green + clover + dead
        logits = torch.cat([green, dead, clover, gdm, total], dim=1)  # [B, 5]
        return {"logits": logits}


# %%
# =============================================================================
# Fast pipeline: ViT 预计算缓存 + head 推理（双进程/双卡并行）
# =============================================================================
def _read_test_unique_images(test_csv: Path) -> tuple[pd.DataFrame, List[str], List[str]]:
    test_df = pd.read_csv(test_csv)
    test_df["image_id"] = test_df["sample_id"].astype(str).str.split("__", n=1).str[0]
    img_df = test_df.drop_duplicates("image_id")[["image_id", "image_path"]].reset_index(drop=True)
    return test_df, img_df["image_id"].tolist(), img_df["image_path"].tolist()


def _split_indices(n: int, parts: int) -> List[np.ndarray]:
    idx = np.arange(n)
    return np.array_split(idx, parts)


def _ensure_cache_memmaps(
    *,
    cache_dir: Path,
    n: int,
    small_tiles: int,
    big_tiles: int,
    feat_dim: int,
) -> tuple[Path, Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    small_path = cache_dir / "tok_small.fp16.mmap"
    big_path = cache_dir / "tok_big.fp16.mmap"
    meta_path = cache_dir / "meta.json"

    # 创建/覆盖文件到目标大小（memmap 依赖文件大小）
    def _init_file(p: Path, shape: tuple[int, ...], dtype: np.dtype):
        mm = np.memmap(p, mode="w+", dtype=dtype, shape=shape)
        mm.flush()
        del mm

    np_dtype = np.float16 if CACHE_DTYPE == "fp16" else np.float32
    _init_file(small_path, (n, small_tiles, feat_dim), np_dtype)
    _init_file(big_path, (n, big_tiles, feat_dim), np_dtype)

    meta = {
        "n": n,
        "small_tiles": small_tiles,
        "big_tiles": big_tiles,
        "feat_dim": feat_dim,
        "small_grid": list(CFG.small_grid),
        "big_grid": list(CFG.big_grid),
        "vit_feat_dim": CFG.vit_feat_dim,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return small_path, big_path, meta_path


def _open_cache_memmaps(
    *,
    small_path: Path,
    big_path: Path,
    n: int,
    small_tiles: int,
    big_tiles: int,
    feat_dim: int,
    mode: str,
) -> tuple[np.memmap, np.memmap]:
    np_dtype = np.float16 if CACHE_DTYPE == "fp16" else np.float32
    tok_small = np.memmap(small_path, mode=mode, dtype=np_dtype, shape=(n, small_tiles, feat_dim))
    tok_big = np.memmap(big_path, mode=mode, dtype=np_dtype, shape=(n, big_tiles, feat_dim))
    return tok_small, tok_big


def _load_image_tensor(image_root: Path, rel_path: str) -> torch.Tensor:
    img_path = image_root / rel_path
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [3,H,W] uint8
    x = x.to(torch.float32) / 255.0
    return x


def _vit_precompute_worker(
    *,
    device_id: int,
    indices: np.ndarray,
    image_paths: List[str],
    image_root: Path,
    small_path: Path,
    big_path: Path,
    n: int,
    small_tiles: int,
    big_tiles: int,
    feat_dim: int,
    vit_batch_size: int,
    dtype_str: str,
) -> None:
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16

    # ViT + transform
    # - thread 后端：可选走“CPU 只加载一次 -> clone 到各 GPU”的共享模型，避免 CPU 同时存在两份 ViT
    # - process 后端：每个进程各自加载（隔离更强）
    if PARALLEL_BACKEND == "thread" and VIT_INIT_CPU_ONCE_CLONE_TO_GPU:
        backbone, backbone_transform = _VIT_THREAD_MODELS[device_id]
        # 确保 dtype/设备正确
        backbone = backbone.to(device=device, dtype=dtype).eval()
    else:
        backbone, backbone_transform = build_vit_backbone(
            vit_name=CFG.vit_name, vit_ckpt=CFG.vit_checkpoint, device=device, dtype=dtype
        )

    tok_small_mm, tok_big_mm = _open_cache_memmaps(
        small_path=small_path,
        big_path=big_path,
        n=n,
        small_tiles=small_tiles,
        big_tiles=big_tiles,
        feat_dim=feat_dim,
        mode="r+",
    )

    backbone.eval()
    with torch.no_grad():
        for start in range(0, len(indices), vit_batch_size):
            batch_idx = indices[start : start + vit_batch_size]
            imgs = [_load_image_tensor(image_root, image_paths[i]) for i in batch_idx.tolist()]
            x = torch.stack(imgs, dim=0).to(device=device, non_blocking=True)  # [B,3,H,W] float32
            # 编码
            # 恢复原版预处理路径（exact_timm）时，这里会走 timm transform；否则走 fast_gpu
            feats_small = encode_tiles_infer(x, backbone, backbone_transform, CFG.small_grid, dtype=dtype)  # [B,Ts,4096]
            feats_big = encode_tiles_infer(x, backbone, backbone_transform, CFG.big_grid, dtype=dtype)  # [B,Tb,4096]
            if CACHE_DTYPE == "fp32":
                tok_small_mm[batch_idx, :, :] = feats_small.detach().to("cpu", torch.float32).numpy()
                tok_big_mm[batch_idx, :, :] = feats_big.detach().to("cpu", torch.float32).numpy()
            else:
                tok_small_mm[batch_idx, :, :] = feats_small.detach().to("cpu", torch.float16).numpy()
                tok_big_mm[batch_idx, :, :] = feats_big.detach().to("cpu", torch.float16).numpy()
            tok_small_mm.flush()
            tok_big_mm.flush()

    del tok_small_mm, tok_big_mm
    # thread 后端下 backbone 是共享的，不在 worker 里 del
    if not (PARALLEL_BACKEND == "thread" and VIT_INIT_CPU_ONCE_CLONE_TO_GPU):
        del backbone
    torch.cuda.empty_cache()


def _load_fold_head_model(device: torch.device, dtype: torch.dtype, fold: int) -> CrossPVT_T2T_MambaHead:
    head_cfg = HFConfig(
        dropout=CFG.dropout,
        hidden_ratio=CFG.hidden_ratio,
        aux_head=CFG.aux_head,
        small_grid=CFG.small_grid,
        big_grid=CFG.big_grid,
        pyramid_dims=CFG.pyramid_dims,
        mamba_depth=CFG.mamba_depth,
        mamba_kernel=CFG.mamba_kernel,
        mobilevit_heads=CFG.mobilevit_heads,
        mobilevit_depth=CFG.mobilevit_depth,
        sra_heads=CFG.sra_heads,
        sra_ratio=CFG.sra_ratio,
        cross_heads=CFG.cross_heads,
        cross_layers=CFG.cross_layers,
        t2t_depth=CFG.t2t_depth,
        vit_name=CFG.vit_name,
        vit_checkpoint=Path("unused"),
        vit_feat_dim=CFG.vit_feat_dim,
        token_embed_dim=CFG.token_embed_dim,
    )
    if HEAD_WEIGHT_DTYPE == "fp32":
        head = CrossPVT_T2T_MambaHead(head_cfg).to(device=device, dtype=torch.float32).eval()
    elif HEAD_WEIGHT_DTYPE == "bf16":
        head = CrossPVT_T2T_MambaHead(head_cfg).to(device=device, dtype=torch.bfloat16).eval()
    else:
        head = CrossPVT_T2T_MambaHead(head_cfg).to(device=device, dtype=torch.float16).eval()

    w = KAGGLE_HEAD_5FOLD_DIR / f"fold{fold}" / "model.safetensors"
    if not w.exists():
        raise FileNotFoundError(f"找不到 fold{fold} 权重: {w}")
    sd = safetensors_load_file(str(w))
    head_sd = {k[len("head.") :]: v for k, v in sd.items() if k.startswith("head.")}
    missing, unexpected = head.load_state_dict(head_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"fold{fold} head load_state_dict 不匹配：missing={missing[:10]} unexpected={unexpected[:10]}")
    return head


@torch.no_grad()
def _head_forward_to_logits(
    head: CrossPVT_T2T_MambaHead,
    tok_small: torch.Tensor,
    tok_big: torch.Tensor,
    small_grid: Tuple[int, int],
) -> torch.Tensor:
    if USE_AUTOCAST_FOR_HEAD and tok_small.is_cuda:
        # Trainer(fp16/bf16) 的 eval 通常会启用 autocast；这里模拟同类行为以减小差异
        with torch.autocast(device_type="cuda", dtype=tok_small.dtype):
            out = head(tok_small, tok_big, small_grid, return_features=False)
    else:
        out = head(tok_small, tok_big, small_grid, return_features=False)
    green = out["green"]
    clover = out["clover"]
    dead = out["dead"]
    gdm = green + clover
    total = green + clover + dead
    return torch.cat([green, dead, clover, gdm, total], dim=1)  # [B,5]


def _head_infer_worker(
    *,
    device_id: int,
    indices: np.ndarray,
    small_path: Path,
    big_path: Path,
    pred_path: Path,
    n: int,
    small_tiles: int,
    big_tiles: int,
    feat_dim: int,
    head_batch_size: int,
    dtype_str: str,
) -> None:
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.float16 if dtype_str == "fp16" else torch.bfloat16

    tok_small_mm, tok_big_mm = _open_cache_memmaps(
        small_path=small_path,
        big_path=big_path,
        n=n,
        small_tiles=small_tiles,
        big_tiles=big_tiles,
        feat_dim=feat_dim,
        mode="r",
    )
    pred_mm = np.memmap(pred_path, mode="r+", dtype=np.float32, shape=(n, 5))

    # 每张卡各加载 5 个 head（你要求的形式）；数据按 indices 分片做 data-parallel
    heads = [_load_fold_head_model(device, dtype, fold) for fold in FOLDS]

    with torch.no_grad():
        for start in range(0, len(indices), head_batch_size):
            batch_idx = indices[start : start + head_batch_size]
            # 缓存读取：fp32 更贴近原始特征；之后再转成 head 需要的 dtype
            if CACHE_DTYPE == "fp32":
                small_np = np.asarray(tok_small_mm[batch_idx, :, :], dtype=np.float32)
                big_np = np.asarray(tok_big_mm[batch_idx, :, :], dtype=np.float32)
                tok_small = torch.from_numpy(small_np).to(device=device, dtype=dtype, non_blocking=True)
                tok_big = torch.from_numpy(big_np).to(device=device, dtype=dtype, non_blocking=True)
            else:
                small_np = np.asarray(tok_small_mm[batch_idx, :, :], dtype=np.float16)
                big_np = np.asarray(tok_big_mm[batch_idx, :, :], dtype=np.float16)
                tok_small = torch.from_numpy(small_np).to(device=device, dtype=dtype, non_blocking=True)
                tok_big = torch.from_numpy(big_np).to(device=device, dtype=dtype, non_blocking=True)

            logits_sum = None
            for head in heads:
                logits = _head_forward_to_logits(head, tok_small, tok_big, CFG.small_grid)
                logits_sum = logits if logits_sum is None else (logits_sum + logits)
            logits_mean = logits_sum / float(len(heads))
            pred_mm[batch_idx, :] = logits_mean.detach().to("cpu", torch.float32).numpy()
            pred_mm.flush()

    del tok_small_mm, tok_big_mm, pred_mm
    del heads
    torch.cuda.empty_cache()


def run_predict_5fold_fast_cached() -> Path:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("FAST_INFER 需要至少 2 张 GPU（当前不可用）")

    test_df, image_ids, image_paths = _read_test_unique_images(TEST_CSV)
    n = len(image_ids)
    small_tiles = CFG.small_grid[0] * CFG.small_grid[1]
    big_tiles = CFG.big_grid[0] * CFG.big_grid[1]
    feat_dim = CFG.vit_feat_dim

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "image_ids.txt").write_text("\n".join(image_ids), encoding="utf-8")

    small_path, big_path, _ = _ensure_cache_memmaps(
        cache_dir=CACHE_DIR, n=n, small_tiles=small_tiles, big_tiles=big_tiles, feat_dim=feat_dim
    )

    # dtype 选择：ViT/head 都用同一个（通常 fp16 更通用）
    dtype = pick_infer_dtype()
    dtype_str = "bf16" if dtype == torch.bfloat16 else "fp16"

    # 1) ViT 预计算（双进程/双卡）
    idx_splits = _split_indices(n, parts=2)
    backend = PARALLEL_BACKEND
    if backend == "process" and _in_notebook():
        print("[WARN] Notebook 环境下 process/spawn 容易 pickle 失败，已自动切换为 thread 后端。")
        backend = "thread"

    # thread 后端下：先 CPU 加载一次 ViT，再 clone 到两张 GPU，避免 CPU 同时存在两份 ViT
    if backend == "thread" and VIT_INIT_CPU_ONCE_CLONE_TO_GPU:
        _init_vit_models_for_thread_backend(GPU_IDS, dtype=dtype)

    vit_jobs = []
    for proc_i, dev in enumerate(GPU_IDS):
        vit_jobs.append(
            (
                _vit_precompute_worker,
                dict(
                    device_id=dev,
                    indices=idx_splits[proc_i],
                    image_paths=image_paths,
                    image_root=IMAGE_ROOT,
                    small_path=small_path,
                    big_path=big_path,
                    n=n,
                    small_tiles=small_tiles,
                    big_tiles=big_tiles,
                    feat_dim=feat_dim,
                    vit_batch_size=VIT_BATCH_SIZE,
                    dtype_str=dtype_str,
                ),
            )
        )
    _run_parallel_jobs(vit_jobs, backend=backend)

    # 2) head 推理（双进程/双卡），每卡各加载 5 个 head，对数据分片
    pred_path = CACHE_DIR / "pred.fp32.mmap"
    pred_mm = np.memmap(pred_path, mode="w+", dtype=np.float32, shape=(n, 5))
    pred_mm.flush()
    del pred_mm

    head_jobs = []
    for proc_i, dev in enumerate(GPU_IDS):
        head_jobs.append(
            (
                _head_infer_worker,
                dict(
                    device_id=dev,
                    indices=idx_splits[proc_i],
                    small_path=small_path,
                    big_path=big_path,
                    pred_path=pred_path,
                    n=n,
                    small_tiles=small_tiles,
                    big_tiles=big_tiles,
                    feat_dim=feat_dim,
                    head_batch_size=HEAD_BATCH_SIZE,
                    dtype_str=dtype_str,
                ),
            )
        )
    _run_parallel_jobs(head_jobs, backend=backend)

    # 3) 聚合写 submission
    pred_mm = np.memmap(pred_path, mode="r", dtype=np.float32, shape=(n, 5))
    pred_5 = np.asarray(pred_mm, dtype=np.float32)
    del pred_mm

    # image_id -> row index
    id2idx = {iid: i for i, iid in enumerate(image_ids)}
    col2j = {c: j for j, c in enumerate(TARGET_COLS)}
    idx_arr = test_df["image_id"].map(id2idx).to_numpy()
    col_arr = test_df["target_name"].map(col2j).to_numpy()
    y = pred_5[idx_arr, col_arr].astype(np.float32)

    sub = pd.DataFrame({"sample_id": test_df["sample_id"].values, "target": y})
    out_csv = OUTPUT_DIR / "submission.csv"
    sub.to_csv(out_csv, index=False)
    return out_csv



# %%
# =============================================================================
# 推理 + 生成 submission.csv（Trainer.predict）
# =============================================================================
def make_submission_from_image_preds(
    *,
    test_csv: Path,
    image_ids: List[str],
    image_pred_5: np.ndarray,
    out_csv: Path,
) -> Path:
    test_df = pd.read_csv(test_csv)
    test_df["image_id"] = test_df["sample_id"].astype(str).str.split("__", n=1).str[0]

    pred_df = pd.DataFrame(image_pred_5, columns=list(TARGET_COLS))
    pred_df["image_id"] = image_ids

    merged = test_df.merge(pred_df, on="image_id", how="left")
    if merged[list(TARGET_COLS)].isna().any().any():
        raise RuntimeError("merge 后存在缺失预测：请检查 image_id 对齐是否一致")

    # 每行根据 target_name 选择对应列
    def pick_row(r):
        return float(r[r["target_name"]])

    merged["target"] = merged.apply(pick_row, axis=1)
    sub = merged[["sample_id", "target"]].copy()
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    return out_csv


def run_predict_5fold() -> Path:
    # 设备选择：默认 head 在 cuda:0；若启用模型并行且 >=2 GPU，则 backbone=cuda:0, head=cuda:1
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 0

    if USE_MODEL_PARALLEL and n_gpu >= 2:
        backbone_device = torch.device(BACKBONE_DEVICE_STR)
        head_device = torch.device(HEAD_DEVICE_STR)
    else:
        # 回退：单卡/CPU
        head_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone_device = head_device

    dtype = pick_infer_dtype()

    # 1) ViT backbone（单独目录加载）
    backbone, backbone_transform = build_vit_backbone(
        vit_name=CFG.vit_name,
        vit_ckpt=CFG.vit_checkpoint,
        device=backbone_device,
        dtype=dtype,
    )

    # 2) dataset（图片级去重）
    ds = BiomassHFTestImageDataset(TEST_CSV, IMAGE_ROOT)

    # 3) TrainingArguments（每折复用；Trainer/model 每折重建以便显存释放）
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "tmp_trainer"),
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none",
        remove_unused_columns=False,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        # 只用单进程单卡（Trainer 内部的 device 仍然是 head_device；backbone 在 forward 里手动搬运）
    )

    # 4) 逐 fold 推理（常驻 1 个 head；推理后释放）
    pred_sum: np.ndarray | None = None
    for fold in FOLDS:
        w = KAGGLE_HEAD_5FOLD_DIR / f"fold{fold}" / "model.safetensors"
        if not w.exists():
            raise FileNotFoundError(f"找不到 fold{fold} 权重: {w}")

        model = HFSingleFoldWrapper(
            cfg=CFG,
            backbone=backbone,
            backbone_transform=backbone_transform,
            dtype=dtype,
            backbone_device=backbone_device,
            head_device=head_device,
        ).to(device=head_device, dtype=dtype)
        model.load_head_from_safetensors(w)

        trainer = Trainer(
            model=model,
            args=args,
            data_collator=data_collator_infer,
        )

        pred = trainer.predict(ds)
        image_pred_5 = pred.predictions
        if isinstance(image_pred_5, (tuple, list)):
            image_pred_5 = image_pred_5[0]
        image_pred_5 = np.asarray(image_pred_5, dtype=np.float32)
        assert image_pred_5.shape[1] == 5, f"预测维度不对: {image_pred_5.shape}"

        if pred_sum is None:
            pred_sum = image_pred_5
        else:
            pred_sum = pred_sum + image_pred_5

        # 释放当前 fold 的 head 显存
        del trainer
        del model
        if head_device.type == "cuda":
            torch.cuda.empty_cache()

    assert pred_sum is not None
    image_pred_5 = pred_sum / float(len(FOLDS))

    out_csv = OUTPUT_DIR / "submission.csv"
    make_submission_from_image_preds(
        test_csv=TEST_CSV,
        image_ids=ds.image_ids,
        image_pred_5=image_pred_5,
        out_csv=out_csv,
    )
    return out_csv


# %%
if __name__ == "__main__":
    # Kaggle 只需要跑这句就能生成 submission.csv
    if FAST_INFER:
        out = run_predict_5fold_fast_cached()
    else:
        out = run_predict_5fold()
    print(f"[OK] saved: {out}")


