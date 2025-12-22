"""
HF Trainer: hold out one State as pseudo-LB, and split the rest 80/20 for train/CV.

- Uses the same head/output-mode rules as train_hf_trainer_state_cv.py.
- Reports CV metrics and pseudo-LB metrics for comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import cv2
import torch
from accelerate import Accelerator
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from train_hf_trainer_state_cv import (
    HFConfig,
    OUTPUT_MODES,
    TARGET_COLS,
    WEIGHTS,
    HFWrapper,
    build_feature_extractor,
)

from train_hf_trainer import add_season_column, build_full_image_transform


N_SPLITS = 5


class BiomassLBHFDataset:
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
        labels = torch.tensor([row[col] for col in TARGET_COLS], dtype=torch.float32)
        is_lb = torch.tensor([row["is_lb"]], dtype=torch.float32)
        labels = torch.cat([labels, is_lb], dim=0)
        return {"pixel_values": img_t, "labels": labels}


def data_collator(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def _weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    w_base = np.asarray([WEIGHTS[c] for c in TARGET_COLS], dtype=np.float32)
    w = np.broadcast_to(w_base, y_true.shape)
    w_flat = w.reshape(-1)
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    y_w = (w_flat * y_true_flat).sum() / w_flat.sum()
    ss_res = (w_flat * (y_true_flat - y_pred_flat) ** 2).sum()
    ss_tot = (w_flat * (y_true_flat - y_w) ** 2).sum()
    return float(1.0 - (ss_res / ss_tot))


def compute_metrics(eval_pred) -> Dict[str, float]:
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    y_pred = np.asarray(preds, dtype=np.float32)
    y_true_full = np.asarray(eval_pred.label_ids, dtype=np.float32)

    y_true = y_true_full[:, : len(TARGET_COLS)]
    is_lb = y_true_full[:, -1] > 0.5
    is_cv = ~is_lb

    metrics: Dict[str, float] = {}
    if is_cv.any():
        metrics["weighted_r2"] = _weighted_r2(y_true[is_cv], y_pred[is_cv])
    if is_lb.any():
        metrics["weighted_r2_lb"] = _weighted_r2(y_true[is_lb], y_pred[is_lb])
    return metrics


def _split_with_stratified_groups(
    meta: pd.DataFrame,
    *,
    seed: int,
    cv_fold: int,
) -> Tuple[pd.Index, pd.Index]:
    meta = meta.copy().reset_index(drop=True)
    meta["fold"] = -1
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold, (_, va_idx) in enumerate(sgkf.split(meta, y=meta["stratify_key"], groups=meta["image_id"])):
            meta.loc[va_idx, "fold"] = fold
    except Exception:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(meta))
        meta.loc[perm, "fold"] = np.arange(len(meta)) % N_SPLITS

    cv_ids = meta.loc[meta["fold"] == cv_fold, "image_id"]
    train_ids = meta.loc[meta["fold"] != cv_fold, "image_id"]
    return train_ids, cv_ids


def split_state_lb_cv(
    df: pd.DataFrame,
    *,
    lb_state: str,
    seed: int,
    cv_fold: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["image_id"] = df["sample_id"].str.split("__").str[0]
    df = add_season_column(df, "Sampling_Date")

    meta = df.drop_duplicates("image_id")[["image_id", "State", "Season"]].copy().reset_index(drop=True)
    meta["State"] = meta["State"].fillna("Unknown")
    meta["Season"] = meta["Season"].fillna("Unknown")
    meta["stratify_key"] = meta["State"].astype(str) + "|" + meta["Season"].astype(str)

    lb_ids = meta.loc[meta["State"] == lb_state, "image_id"]
    remain_meta = meta.loc[meta["State"] != lb_state].reset_index(drop=True)
    train_ids, cv_ids = _split_with_stratified_groups(remain_meta, seed=seed, cv_fold=cv_fold)

    wide = (
        df.pivot_table(index=["image_id", "image_path"], columns="target_name", values="target")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    train_df = wide[wide["image_id"].isin(train_ids)].reset_index(drop=True)
    cv_df = wide[wide["image_id"].isin(cv_ids)].reset_index(drop=True)
    lb_df = wide[wide["image_id"].isin(lb_ids)].reset_index(drop=True)
    return train_df, cv_df, lb_df


class LBCVTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is not None and labels.size(-1) > len(TARGET_COLS):
            inputs = dict(inputs)
            inputs["labels"] = labels[:, : len(TARGET_COLS)]
        return super().compute_loss(model, inputs, return_outputs)


@dataclass
class LBCVConfig:
    train_csv: Path = Path("csiro-biomass/train.csv")
    image_dir: Path = Path("csiro-biomass")
    output_dir: Path = Path("runs/hf_trainer_state_lb_cv")
    epochs: int = 5
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 1e-5
    weight_decay: float = 1e-4
    dropout: float = 0.1
    hidden_ratio: float = 0.35
    aux_head: bool = True
    bf16: bool = False
    output_mode: str = "exp1"
    nonneg_clamp: bool = True
    lb_state: str = "NSW"
    cv_fold: int = 0
    cv_folds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    seed: int = 42


def parse_args():
    ap = argparse.ArgumentParser(description="Train with CV + pseudo-LB (state holdout)")
    ap.add_argument("--train-csv", type=Path, default=LBCVConfig.train_csv)
    ap.add_argument("--image-dir", type=Path, default=LBCVConfig.image_dir)
    ap.add_argument("--output-dir", type=Path, default=LBCVConfig.output_dir)
    ap.add_argument("--epochs", type=int, default=LBCVConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=LBCVConfig.batch_size)
    ap.add_argument("--num-workers", type=int, default=LBCVConfig.num_workers)
    ap.add_argument("--lr", type=float, default=LBCVConfig.lr)
    ap.add_argument("--weight-decay", type=float, default=LBCVConfig.weight_decay)
    ap.add_argument("--dropout", type=float, default=LBCVConfig.dropout)
    ap.add_argument("--hidden-ratio", type=float, default=LBCVConfig.hidden_ratio)
    ap.add_argument("--output-mode", type=str, default=LBCVConfig.output_mode, choices=sorted(OUTPUT_MODES.keys()))
    ap.add_argument("--lb-state", type=str, default=LBCVConfig.lb_state)
    ap.add_argument("--cv-fold", type=str, default=str(LBCVConfig.cv_fold))
    ap.add_argument("--cv-folds", type=str, default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=LBCVConfig.seed)
    ap.add_argument("--no-nonneg-clamp", action="store_true")
    ap.add_argument("--no-aux", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device
    if "," in args.cv_fold:
        cv_folds = tuple(int(x) for x in args.cv_fold.split(",") if x.strip() != "")
    else:
        cv_folds = tuple(int(x) for x in str(args.cv_folds).split(",") if x.strip() != "")

    cfg = LBCVConfig(
        train_csv=args.train_csv,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_ratio=args.hidden_ratio,
        aux_head=not args.no_aux,
        bf16=args.bf16,
        output_mode=args.output_mode,
        nonneg_clamp=not args.no_nonneg_clamp,
        lb_state=args.lb_state,
        cv_fold=int(args.cv_fold.split(",")[0]),
        cv_folds=cv_folds,
        seed=args.seed,
    )

    df = pd.read_csv(cfg.train_csv)
    backbone, backbone_transform = build_feature_extractor(HFConfig(device=device))

    all_metrics = []
    for cv_fold in cfg.cv_folds:
        train_df, cv_df, lb_df = split_state_lb_cv(df, lb_state=cfg.lb_state, seed=cfg.seed, cv_fold=cv_fold)
        print(
            f"\n===== CV Fold {cv_fold} =====\n"
            f"Train rows: {len(train_df)}, CV rows: {len(cv_df)}, "
            f"Pseudo-LB rows ({cfg.lb_state}): {len(lb_df)}"
        )

        train_df = train_df.copy()
        cv_df = cv_df.copy()
        lb_df = lb_df.copy()
        train_df["is_lb"] = 0.0
        cv_df["is_lb"] = 0.0
        lb_df["is_lb"] = 1.0
        eval_df = pd.concat([cv_df, lb_df], ignore_index=True)

        train_ds = BiomassLBHFDataset(train_df, cfg.image_dir, train=True)
        eval_ds = BiomassLBHFDataset(eval_df, cfg.image_dir, train=False)

        model = HFWrapper(
            HFConfig(
                train_csv=cfg.train_csv,
                image_dir=cfg.image_dir,
                output_dir=cfg.output_dir,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                epochs=cfg.epochs,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                dropout=cfg.dropout,
                hidden_ratio=cfg.hidden_ratio,
                aux_head=cfg.aux_head,
                bf16=cfg.bf16,
                output_mode=cfg.output_mode,
                nonneg_clamp=cfg.nonneg_clamp,
            ),
            backbone,
            backbone_transform,
        )

        args_tr = TrainingArguments(
            output_dir=str(cfg.output_dir / f"{cfg.output_mode}_{cfg.lb_state}_cv{cv_fold}"),
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
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="weighted_r2",
            greater_is_better=True,
            report_to="tensorboard",
        )

        trainer = LBCVTrainer(
            model=model,
            args=args_tr,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
        )

        trainer.train()
        metrics = trainer.evaluate()
        cv_metrics = {"weighted_r2": metrics.get("weighted_r2")}
        lb_metrics = {"weighted_r2_lb": metrics.get("weighted_r2_lb")}
        trainer.log({**cv_metrics, **lb_metrics})
        all_metrics.append((cv_fold, cv_metrics, lb_metrics))
        print(f"CV metrics: {cv_metrics}")
        print(f"Pseudo-LB metrics: {lb_metrics}")

    if all_metrics:
        cv_scores = [float(m[1].get("weighted_r2", -1.0)) for m in all_metrics]
        lb_scores = [float(m[2].get("weighted_r2_lb", -1.0)) for m in all_metrics]
        print(
            f"\nMean CV weighted_r2: {np.mean(cv_scores):.5f} | "
            f"Mean LB weighted_r2: {np.mean(lb_scores):.5f}"
        )


if __name__ == "__main__":
    main()
