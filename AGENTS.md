# Repository Guidelines

## Project Structure & Module Organization
- `csiro-biomass/`: Kaggle data drop (train/test CSVs plus image folders); treat as read-only input.  
- `evaluation.py`: weighted R² metric and a helper to generate solution/submission pairs from `train.csv` for sanity checks.  
- `data_enhance.py`: Albumentations pipelines (train/val) for tile crops and normalization.  
- `data_feature_inspector/inspector.ipynb`: exploratory notebook for feature rules and distributions.  
- `timm/vit_7b_patch16_dinov3.lvd1689m/`: pretrained weight snapshot; keep the path stable for loaders.

## Build, Test, and Development Commands
- Verify metric implementation: `python evaluation.py --generate-from-train csiro-biomass/train.csv` (writes `solution.csv` and `submission.csv`, then evaluates).  
- Score a submission: `python evaluation.py --solution path/to/solution.csv --submission path/to/submission.csv`.  
- Notebook work: `jupyter lab data_feature_inspector/inspector.ipynb` (avoid committing large outputs).  
- Augmentations live in `data_enhance.py`; import the composed transforms rather than redefining them in training scripts.

## Coding Style & Naming Conventions
- Python 3.10+: 4-space indent, type hints for public functions, prefer `Path` over string paths, and vectorized pandas/numpy operations.  
- Keep augmentation and config constants in ALL_CAPS (see `MEAN`, `STD`, `SIZE`).  
- Functions: snake_case (`generate_solution_and_submission_from_train`), modules: snake_case files, experiment plans in kebab/underscore markdown.  
- Localize heavy-weight assets (models, data) under clearly named directories; avoid hard-coding absolute paths.

## Testing & Validation
- No formal test suite yet; add `tests/` with `test_*.py` when adding logic.  
- For metric changes, create small CSV fixtures and assert exact weighted R² outputs.  
- When touching augmentations, log deterministic seeds and run a short smoke pass over a handful of tiles to catch shape/type regressions.

## Commit & Pull Request Guidelines
- Commit messages: concise present tense; include scope (e.g., `eval: guard missing target weights`).  
- PRs should note: problem statement, data dependencies (paths/weights), commands run, and before/after metrics or screenshots for notebooks.  
- Avoid committing generated CSVs beyond minimal fixtures; keep large artifacts in external storage with a README pointer.  
- Run the relevant commands above before opening a PR and summarize the results in the description.

## Data & Security Tips
- Treat `csiro-biomass/train/` as immutable input; never rewrite it in-place.  
- Do not upload proprietary weights (`timm/...`) or raw data to public remotes.  
- Strip notebooks of outputs when they contain sensitive paths or metrics not meant for release.
