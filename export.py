# %%
# =============================================================================
# （可选）本地导出：从 runs/hf_trainer/fold*/checkpoint-*/trainer_state.json 自动找 best ckpt 并按 A 格式复制
# -----------------------------------------------------------------------------
# Kaggle 上一般不需要运行这一段（因为你会先把 export_models 做成 Kaggle dataset）
# =============================================================================
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

def _read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_bf16(model_path: Path) -> None:
    tensors = load_file(str(model_path))
    needs_convert = any(t.dtype != torch.bfloat16 for t in tensors.values())
    if not needs_convert:
        return
    converted = {k: v.to(dtype=torch.bfloat16) for k, v in tensors.items()}
    save_file(converted, str(model_path))

def _resolve_best_checkpoint(best_rel: str, runs_root: Path, exp_dir: Path | None) -> Path:
    best_ckpt = Path(best_rel)
    if best_ckpt.is_absolute():
        return best_ckpt
    candidates = [
        Path(best_rel),
        exp_dir / best_rel if exp_dir is not None else None,
        runs_root / best_rel,
        runs_root.parent.parent / best_rel,
    ]
    for cand in candidates:
        if cand is not None and cand.exists():
            return cand.resolve()
    return best_ckpt.resolve()

def export_best_5fold_checkpoints_A(
    *,
    runs_root: Path,
    export_root: Path,
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    ensure_bf16: bool = True,
) -> Path:
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    meta = {"format": "A", "folds": list(folds), "source_runs_root": str(runs_root)}

    for fold in folds:
        fold_dir = runs_root / f"fold{fold}"
        ckpts = sorted([p for p in fold_dir.glob("checkpoint-*") if p.is_dir()], key=lambda p: int(p.name.split("-")[1]))
        if not ckpts:
            raise FileNotFoundError(f"未找到 checkpoints: {fold_dir}")
        # 取最后一个 checkpoint 的 trainer_state.json（里面包含 best_model_checkpoint）
        last_ckpt = ckpts[-1]
        ts_path = last_ckpt / "trainer_state.json"
        if not ts_path.exists():
            raise FileNotFoundError(f"trainer_state.json 不存在: {ts_path}")

        ts = _read_json(ts_path)
        best_rel = ts.get("best_model_checkpoint")
        if not best_rel:
            raise ValueError(f"trainer_state.json 缺少 best_model_checkpoint: {ts_path}")

        best_ckpt = Path(best_rel)
        if not best_ckpt.is_absolute():
            # best_rel 通常是相对路径（如 runs/hf_trainer/fold0/checkpoint-1512）
            best_ckpt = (runs_root.parent.parent / best_ckpt).resolve()

        best_model = best_ckpt / "model.safetensors"
        if not best_model.exists():
            raise FileNotFoundError(f"best model.safetensors 不存在: {best_model}")

        out_dir = export_root / f"fold{fold}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "model.safetensors"
        shutil.copy2(best_model, out_path)
        if ensure_bf16:
            _ensure_bf16(out_path)

        meta[f"fold{fold}"] = {
            "best_metric": ts.get("best_metric"),
            "best_global_step": ts.get("best_global_step"),
            "best_checkpoint": str(best_ckpt),
            "exported": str(out_dir / "model.safetensors"),
        }

    meta_path = export_root / "export_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return export_root

def export_best_checkpoints_by_exp(
    *,
    runs_root: Path,
    export_root: Path,
    folds: Tuple[int, ...] = (0, 1, 2, 3, 4),
    exp_prefix: str = "exp",
    ensure_bf16: bool = True,
) -> Dict[str, Path]:
    runs_root = Path(runs_root)
    export_root = Path(export_root)
    export_root.mkdir(parents=True, exist_ok=True)

    exp_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith(exp_prefix)])
    exp_fold_pattern = re.compile(rf"^({re.escape(exp_prefix)}\d+)_fold_(\d+)$")
    exp_fold_dirs = [p for p in exp_dirs if exp_fold_pattern.match(p.name)]

    if not exp_dirs:
        out = export_best_5fold_checkpoints_A(
            runs_root=runs_root,
            export_root=export_root,
            folds=folds,
            ensure_bf16=ensure_bf16,
        )
        return {runs_root.name: out}

    if exp_fold_dirs:
        groups: Dict[str, Dict[int, Path]] = {}
        for exp_dir in exp_fold_dirs:
            match = exp_fold_pattern.match(exp_dir.name)
            if not match:
                continue
            exp_name = match.group(1)
            fold = int(match.group(2))
            groups.setdefault(exp_name, {})[fold] = exp_dir

        results: Dict[str, Path] = {}
        for exp_name, fold_dirs in sorted(groups.items()):
            exp_export_root = export_root / exp_name
            exp_export_root.mkdir(parents=True, exist_ok=True)
            meta = {"format": "A", "folds": sorted(fold_dirs.keys()), "source_runs_root": str(runs_root)}

            for fold, fold_dir in sorted(fold_dirs.items()):
                ckpts = sorted(
                    [p for p in fold_dir.glob("checkpoint-*") if p.is_dir()],
                    key=lambda p: int(p.name.split("-")[1]),
                )
                if not ckpts:
                    raise FileNotFoundError(f"未找到 checkpoints: {fold_dir}")
                last_ckpt = ckpts[-1]
                ts_path = last_ckpt / "trainer_state.json"
                if not ts_path.exists():
                    raise FileNotFoundError(f"trainer_state.json 不存在: {ts_path}")

                ts = _read_json(ts_path)
                best_rel = ts.get("best_model_checkpoint")
                if not best_rel:
                    raise ValueError(f"trainer_state.json 缺少 best_model_checkpoint: {ts_path}")

                best_ckpt = _resolve_best_checkpoint(best_rel, runs_root, fold_dir)
                best_model = best_ckpt / "model.safetensors"
                if not best_model.exists():
                    raise FileNotFoundError(f"best model.safetensors 不存在: {best_model}")

                out_dir = exp_export_root / f"fold{fold}"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / "model.safetensors"
                shutil.copy2(best_model, out_path)
                if ensure_bf16:
                    _ensure_bf16(out_path)

                meta[f"fold{fold}"] = {
                    "best_metric": ts.get("best_metric"),
                    "best_global_step": ts.get("best_global_step"),
                    "best_checkpoint": str(best_ckpt),
                    "exported": str(out_dir / "model.safetensors"),
                }

            meta_path = exp_export_root / "export_meta.json"
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            results[exp_name] = exp_export_root
        return results

    results: Dict[str, Path] = {}
    for exp_dir in exp_dirs:
        exp_export_root = export_root / exp_dir.name
        exp_export_root.mkdir(parents=True, exist_ok=True)
        meta = {"format": "A", "folds": list(folds), "source_runs_root": str(exp_dir)}

        for fold in folds:
            fold_dir = exp_dir / f"fold{fold}"
            ckpts = sorted(
                [p for p in fold_dir.glob("checkpoint-*") if p.is_dir()],
                key=lambda p: int(p.name.split("-")[1]),
            )
            if not ckpts:
                raise FileNotFoundError(f"未找到 checkpoints: {fold_dir}")
            last_ckpt = ckpts[-1]
            ts_path = last_ckpt / "trainer_state.json"
            if not ts_path.exists():
                raise FileNotFoundError(f"trainer_state.json 不存在: {ts_path}")

            ts = _read_json(ts_path)
            best_rel = ts.get("best_model_checkpoint")
            if not best_rel:
                raise ValueError(f"trainer_state.json 缺少 best_model_checkpoint: {ts_path}")

            best_ckpt = _resolve_best_checkpoint(best_rel, runs_root, exp_dir)
            best_model = best_ckpt / "model.safetensors"
            if not best_model.exists():
                raise FileNotFoundError(f"best model.safetensors 不存在: {best_model}")

            out_dir = exp_export_root / f"fold{fold}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "model.safetensors"
            shutil.copy2(best_model, out_path)
            if ensure_bf16:
                _ensure_bf16(out_path)

            meta[f"fold{fold}"] = {
                "best_metric": ts.get("best_metric"),
                "best_global_step": ts.get("best_global_step"),
                "best_checkpoint": str(best_ckpt),
                "exported": str(out_dir / "model.safetensors"),
            }

        meta_path = exp_export_root / "export_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        results[exp_dir.name] = exp_export_root
    return results

if __name__ == "__main__":  
    export_best_checkpoints_by_exp(runs_root=Path("runs/hf_trainer_auxtest_log1p"), export_root=Path("runs/hf_trainer_aux_test_exp_logp"))
