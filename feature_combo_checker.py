from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path


ALL_FEATURES = (
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
)


def parse_inferenced_feature_combos(md_path: Path) -> set[frozenset[str]]:
    """
    Parse `model inferenced feature: ...` lines in experiments_plan.md.

    Returns:
        Set of combos; each combo is a frozenset of feature names.
    """
    text = md_path.read_text(encoding="utf-8")
    combos: set[frozenset[str]] = set()

    # Accept extra spaces; stop at end-of-line
    pattern = re.compile(r"^\s*model inferenced feature:\s*(.+?)\s*$", re.MULTILINE)
    for m in pattern.finditer(text):
        raw = m.group(1)
        items = [x.strip() for x in raw.split(",") if x.strip()]
        combos.add(frozenset(items))

    return combos


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    md_path = repo_root / "experiments_plan.md"

    existing = parse_inferenced_feature_combos(md_path)
    universe = {frozenset(c) for c in combinations(ALL_FEATURES, 3)}
    missing = sorted((universe - existing), key=lambda s: tuple(sorted(s)))

    print(f"ALL_FEATURES (n={len(ALL_FEATURES)}): {', '.join(ALL_FEATURES)}")
    print(f"C(5,3) universe size: {len(universe)}")
    print(f"Existing combos in {md_path.name}: {len(existing)}")
    print(f"Missing combos: {len(missing)}")
    print()

    if missing:
        print("=== Missing combos (3 features each) ===")
        for combo in missing:
            print(", ".join(sorted(combo)))


if __name__ == "__main__":
    main()







