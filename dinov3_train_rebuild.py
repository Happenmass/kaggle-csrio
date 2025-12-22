"""
Reconstructed DINOv3-style self-supervised trainer (student/teacher EMA + multi-crop).

This script is a compact rewrite of dinov3/train/train.py tailored for image-folder
pretraining. Launch with torchrun for multi-GPU:

  torchrun --nproc_per_node=4 dinov3_train_rebuild.py --image-root csiro-biomass/train
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path
from typing import List

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets.folder import default_loader

from data_enhance import MEAN, STD


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_schedule(base: float, final: float, total_steps: int, warmup_steps: int = 0, start_warmup: float = 0.0):
    schedule = []
    for i in range(total_steps):
        if i < warmup_steps:
            val = start_warmup + (base - start_warmup) * i / max(1, warmup_steps)
        else:
            t = (i - warmup_steps) / max(1, total_steps - warmup_steps)
            val = final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * t))
        schedule.append(val)
    return schedule


def make_albu_crop(size: int, scale: tuple[float, float], solarize: bool = False) -> A.Compose:
    """Albumentations pipeline inspired by data_enhance.py, with adjustable crop scale."""
    size_tuple = (size, size)
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=size_tuple,
                scale=scale,
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                ],
                p=0.8,
            ),
            A.RandomBrightnessContrast(0.15, 0.15, p=0.6),
            A.HueSaturationValue(8, 12, 8, p=0.4),
            A.RandomGamma(gamma_limit=(85, 115), p=0.25),
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(55, 95)),
                    A.GaussNoise(std_range=(0.02, 0.10)),
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.MotionBlur(blur_limit=3),
                ],
                p=0.30,
            ),
            A.Solarize(p=0.2 if solarize else 0.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


class MultiCrop:
    """Multi-crop wrapper using Albumentations transforms aligned with data_enhance."""

    def __init__(self, g_size=256, l_size=112, n_global=2, n_local=6, g_scale=(0.55, 1.0), l_scale=(0.05, 0.32)):
        self.n_global = n_global
        self.n_local = n_local
        self.global_tfm = make_albu_crop(g_size, scale=g_scale, solarize=True)
        self.local_tfm = make_albu_crop(l_size, scale=l_scale, solarize=False)

    def __call__(self, img):
        # Convert PIL -> numpy (BGR) for albumentations
        np_img = np.array(img)
        if np_img.ndim == 2:  # grayscale safeguard
            np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        else:
            np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        crops = [self.global_tfm(image=np_img)["image"] for _ in range(self.n_global)]
        crops.extend(self.local_tfm(image=np_img)["image"] for _ in range(self.n_local))
        return crops


class LoRALinear(nn.Module):
    """Wrap a Linear layer with a LoRA low-rank update."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / float(r)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        # freeze base weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        x_d = self.dropout(x)
        delta = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return result + self.scaling * delta

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias


class UnlabeledFolder(Dataset):
    def __init__(self, root: str | Path, transform):
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"{root} not found")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
        if not self.files:
            raise ValueError(f"No images found under {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = default_loader(self.files[idx])
        return self.transform(img)


def collate_multi_crop(batch: List[List[torch.Tensor]]):
    n_crops = len(batch[0])
    return [torch.stack([sample[i] for sample in batch]) for i in range(n_crops)]


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 4096, out_dim: int = 65536, nlayers: int = 3, norm_last_layer=True):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (nlayers - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            layers.append(nn.GELU())
        layers.append(nn.Linear(dims[-2], dims[-1], bias=False))
        self.mlp = nn.Sequential(*layers)
        self.norm_last_layer = norm_last_layer
        if norm_last_layer:
            self.last_norm = nn.utils.weight_norm(self.mlp[-1], dim=1)
            self.last_norm.weight_g.data.fill_(1.0)
        else:
            self.last_norm = self.mlp[-1]

    def forward(self, x):
        x = self.mlp[:-1](x)
        x = F.normalize(x, dim=-1)
        x = self.last_norm(x)
        return F.normalize(x, dim=-1)


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, momentum: float):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


def dino_loss(student_outs, teacher_outs, center, teacher_temp: float, student_temp: float) -> torch.Tensor:
    total_loss = 0.0
    n_terms = 0
    for t_out in teacher_outs:
        t_prob = F.softmax((t_out - center) / teacher_temp, dim=-1)
        for s_out in student_outs:
            s_logprob = F.log_softmax(s_out / student_temp, dim=-1)
            total_loss += -(t_prob * s_logprob).sum(dim=-1).mean()
            n_terms += 1
    return total_loss / max(1, n_terms)


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world_size = 0, 1
    return rank, world_size


def inject_lora(model: nn.Module, r: int, alpha: int, dropout: float, target_suffix="qkv"):
    """Replace Linear layers whose name endswith target_suffix with LoRA-wrapped versions."""
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name.endswith(target_suffix):
            parent_path = name.split(".")[:-1]
            attr_name = name.split(".")[-1]
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)
            lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_layer)
            replaced.append(name)
    if not replaced:
        raise RuntimeError(f"No Linear layers ending with '{target_suffix}' found for LoRA injection.")
    return replaced


def freeze_non_lora(model: nn.Module):
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def lora_state_dict(model: nn.Module):
    return {k: v.cpu() for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def main():
    ap = argparse.ArgumentParser(description="Rebuilt DINOv3 pretraining loop.")
    ap.add_argument("--image-root", default="csiro-biomass/train", help="Unlabeled image folder.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size.")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--min-lr", type=float, default=1.5e-6)
    ap.add_argument("--weight-decay", type=float, default=0.04)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--warmup-teacher-temp", type=float, default=0.04)
    ap.add_argument("--teacher-temp", type=float, default=0.07)
    ap.add_argument("--student-temp", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.994)
    ap.add_argument("--final-momentum", type=float, default=1.0)
    ap.add_argument("--global-size", type=int, default=256)
    ap.add_argument("--local-size", type=int, default=112)
    ap.add_argument("--n-local", type=int, default=8)
    ap.add_argument("--out-dim", type=int, default=65536)
    ap.add_argument("--head-hidden", type=int, default=4096)
    ap.add_argument("--norm-last-layer", action="store_true")
    ap.add_argument("--model", default="vit_7b_patch16_dinov3.lvd1689m")
    ap.add_argument("--checkpoint-path", default="timm/vit_7b_patch16_dinov3.lvd1689m/model.safetensors")
    ap.add_argument("--output-dir", default="dino_rebuild_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clip-grad", type=float, default=3.0)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--amp-dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    rank, world_size = init_distributed()
    main_process = rank == 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    transform = MultiCrop(
        g_size=args.global_size,
        l_size=args.local_size,
        n_global=2,
        n_local=args.n_local,
    )
    dataset = UnlabeledFolder(args.image_root, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_multi_crop,
    )

    student_backbone = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=0,
        global_pool="token",
        checkpoint_path=args.checkpoint_path,
    )
    replaced = inject_lora(student_backbone, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, target_suffix="qkv")
    if main_process:
        print(f"LoRA applied to: {replaced}")
    freeze_non_lora(student_backbone)
    teacher_backbone = copy.deepcopy(student_backbone)
    for p in teacher_backbone.parameters():
        p.requires_grad = False
    embed_dim = student_backbone.num_features
    student_head = DINOHead(
        embed_dim,
        hidden_dim=args.head_hidden,
        out_dim=args.out_dim,
        nlayers=3,
        norm_last_layer=args.norm_last_layer,
    )
    teacher_head = copy.deepcopy(student_head)
    for p in teacher_head.parameters():
        p.requires_grad = False

    student_backbone = student_backbone.to(device)
    teacher_backbone = teacher_backbone.to(device)
    student_head = student_head.to(device)
    teacher_head = teacher_head.to(device)

    if args.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
        scaler = None  # bf16 不需要 GradScaler
    elif args.amp_dtype == "fp16":
        amp_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    else:
        amp_dtype = None
        scaler = None

    if world_size > 1:
        student_backbone = DDP(student_backbone, device_ids=[rank], find_unused_parameters=False)
        student_head = DDP(student_head, device_ids=[rank], find_unused_parameters=False)

    params = [p for p in student_backbone.parameters() if p.requires_grad] + list(student_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    total_steps = args.epochs * len(loader)
    lr_schedule = cosine_schedule(args.lr, args.min_lr, total_steps, warmup_steps=args.warmup_epochs * len(loader), start_warmup=0.0)
    wd_schedule = cosine_schedule(args.weight_decay, args.weight_decay, total_steps)
    momentum_schedule = cosine_schedule(args.momentum, args.final_momentum, total_steps)
    teacher_temp_schedule = cosine_schedule(
        args.teacher_temp,
        args.teacher_temp,
        total_steps,
        warmup_steps=args.warmup_epochs * len(loader),
        start_warmup=args.warmup_teacher_temp,
    )

    center = torch.zeros(1, args.out_dim, device=device)
    output_dir = Path(args.output_dir)
    if main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        student_backbone.train()
        student_head.train()
        running_loss = 0.0
        for it, crops in enumerate(loader):
            global_crops = [c.to(device, non_blocking=True) for c in crops[:2]]
            local_crops = [c.to(device, non_blocking=True) for c in crops[2:]]
            optimizer.param_groups[0]["lr"] = lr_schedule[step]
            optimizer.param_groups[0]["weight_decay"] = wd_schedule[step]
            teacher_temp = teacher_temp_schedule[step]
            momentum = momentum_schedule[step]

            optimizer.zero_grad(set_to_none=True)
            autocast_context = (
                torch.amp.autocast
                if device.type == "cuda"
                else torch.cpu.amp.autocast
            )
            with autocast_context(device_type="cuda" if device.type == "cuda" else "cpu", enabled=args.amp_dtype != "fp32", dtype=amp_dtype):
                student_outs = [student_head(student_backbone(c)) for c in global_crops + local_crops]
                with torch.no_grad():
                    teacher_outs = [teacher_head(teacher_backbone(c)) for c in global_crops]
                loss = dino_loss(student_outs, teacher_outs, center, teacher_temp=teacher_temp, student_temp=args.student_temp)
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                optimizer.step()

            update_ema(student_backbone.module if isinstance(student_backbone, DDP) else student_backbone, teacher_backbone, momentum)
            update_ema(student_head.module if isinstance(student_head, DDP) else student_head, teacher_head, momentum)

            with torch.no_grad():
                batch_center = torch.cat(teacher_outs, dim=0).mean(dim=0, keepdim=True)
                if dist.is_initialized():
                    dist.all_reduce(batch_center)
                    batch_center /= world_size
                center = center * 0.9 + batch_center * 0.1

            running_loss += loss.item()
            if main_process and (it + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                print(
                    f"Epoch {epoch+1}/{args.epochs} Iter {it+1}/{len(loader)} "
                    f"loss={avg_loss:.4f} lr={lr_schedule[step]:.2e} mom={momentum:.4f} t_temp={teacher_temp:.3f}"
                )
                running_loss = 0.0
            step += 1

        if main_process:
            base_student = student_backbone.module if isinstance(student_backbone, DDP) else student_backbone
            adapter_dir = output_dir / f"lora_epoch_{epoch+1}"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "lora": lora_state_dict(base_student),
                    "config": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
                },
                adapter_dir / "lora_adapters.pt",
            )
            ckpt = {
                "epoch": epoch + 1,
                "student_head": (student_head.module if isinstance(student_head, DDP) else student_head).state_dict(),
                "teacher": teacher_backbone.state_dict(),
                "teacher_head": teacher_head.state_dict(),
                "center": center,
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "args": vars(args),
            }
            torch.save(ckpt, adapter_dir / "extra_state.pt")
            print(f"Saved LoRA adapter + head to {adapter_dir}")


if __name__ == "__main__":
    main()
