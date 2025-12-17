import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WEIGHTS = {
    "Dry_Green_g": 0.1,
    "Dry_Dead_g": 0.1,
    "Dry_Clover_g": 0.1,
    "GDM_g": 0.2,
    "Dry_Total_g": 0.5,
}


def _ensure_cols(df: pd.DataFrame, *, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}，当前列={list(df.columns)}")


def generate_solution_and_submission_from_train(
    train_csv: str | Path,
    *,
    solution_csv: str | Path,
    submission_csv: str | Path,
) -> tuple[Path, Path]:
    train_csv = Path(train_csv)
    sol_path = Path(solution_csv)
    sub_path = Path(submission_csv)

    train = pd.read_csv(train_csv)
    _ensure_cols(train, required=["sample_id", "target"], name=str(train_csv))

    out = train[["sample_id", "target"]].copy()
    out.to_csv(sol_path, index=False)
    out.to_csv(sub_path, index=False)
    return sol_path, sub_path


def evaluate(solution_csv: str | Path, submission_csv: str | Path) -> float:
    sub = pd.read_csv(submission_csv)  # columns: sample_id, target (pred)
    sol = pd.read_csv(solution_csv)  # columns: sample_id, target (true)
    _ensure_cols(sub, required=["sample_id", "target"], name=str(submission_csv))
    _ensure_cols(sol, required=["sample_id", "target"], name=str(solution_csv))

    sub = sub.rename(columns={"target": "y_pred"})
    sol = sol.rename(columns={"target": "y_true"})

    assert sub["sample_id"].is_unique, "submission 里 sample_id 有重复"
    assert sol["sample_id"].is_unique, "solution 里 sample_id 有重复"

    df = sol.merge(sub, on="sample_id", how="left")

    miss = df["y_pred"].isna().sum()
    assert miss == 0, f"submission 缺少 {miss} 行预测（有些 sample_id 没提交）"

    df["target_name"] = df["sample_id"].str.split("__", n=1).str[1]
    if df["target_name"].isna().any():
        raise ValueError("sample_id 未包含 '__'，无法解析 target_name")

    w = df["target_name"].map(WEIGHTS)
    missing_w = df.loc[w.isna(), "target_name"].unique().tolist()
    if missing_w:
        raise ValueError(f"存在未配置权重的 target_name: {missing_w}")
    w = w.astype(float).to_numpy()

    y = df["y_true"].to_numpy(dtype=float)
    yhat = df["y_pred"].to_numpy(dtype=float)

    y_w = (w * y).sum() / w.sum()
    ss_res = (w * (y - yhat) ** 2).sum()
    ss_tot = (w * (y - y_w) ** 2).sum()
    r2_w = 1.0 - ss_res / ss_tot

    df["w"] = w
    df["res_w"] = df["w"] * (df["y_true"] - df["y_pred"]) ** 2
    df["tot_w"] = df["w"] * (df["y_true"] - y_w) ** 2
    summary = df.groupby("target_name")[["res_w", "tot_w"]].sum()
    total_res = float(summary["res_w"].sum())
    summary["res_share"] = 0.0 if total_res == 0.0 else (summary["res_w"] / total_res)
    summary["tot_share"] = summary["tot_w"] / summary["tot_w"].sum()
    summary = summary.sort_values("res_share", ascending=False)

    print("weighted R2 =", r2_w)
    print(summary)
    return float(r2_w)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--solution", default="solution.csv", help="ground truth solution csv")
    p.add_argument("--submission", default="submission.csv", help="prediction submission csv")
    p.add_argument(
        "--generate-from-train",
        default=None,
        metavar="TRAIN_CSV",
        help="从 train.csv 生成 solution/submission（两者相同，用于验证评估代码）",
    )
    args = p.parse_args()

    if args.generate_from_train:
        sol_path, sub_path = generate_solution_and_submission_from_train(
            args.generate_from_train,
            solution_csv=args.solution,
            submission_csv=args.submission,
        )
        print(f"generated: {sol_path} , {sub_path}")

    evaluate(args.solution, args.submission)


if __name__ == "__main__":
    main()
