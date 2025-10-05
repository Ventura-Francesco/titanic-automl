# Titanic AutoML – Adaptive Hyperparameter & Threshold Optimization

> **Adaptive, explainable, demo‑friendly ML pipeline for the Kaggle Titanic dataset**  
> featuring dynamic hyperparameter expansion, out‑of‑fold (OOF) decision threshold tuning, latency benchmarking, and fully automated CI tests.

<p align="center">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="status"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="license"/>
  <img src="https://img.shields.io/badge/python-3.11+-yellow" alt="python"/>
  <img src="https://img.shields.io/badge/tests-passing-success" alt="tests"/>
  <img src="https://img.shields.io/badge/coverage-TBD-lightgrey" alt="coverage"/>
</p>

---

## 🚀 Why This Project?

Traditional grid searches waste time exploring unpromising regions, and a fixed probability threshold (0.5) rarely maximizes accuracy or business alignment.  
This repository implements a **median‑guided adaptive hyperparameter search** that *expands only where the model shows statistically meaningful frontier improvements*, and then **optimizes the classification threshold using OOF predictions** to squeeze additional performance out of the chosen model.

Result: A robust, reproducible pipeline you can demo live in under 3 minutes (demo mode) *and* scale up (full mode) for stronger leaderboard performance or research.

---

## ✨ Key Features

| Category | Feature | Description |
|----------|---------|-------------|
| Search | Adaptive median‑based expansion | Expands parameter grids only at performance boundaries surpassing a delta threshold. |
| Robustness | Int/None coercion hardening | Safely converts float/int/`None` (e.g. `max_depth`) & rejects NaN leaks. |
| Thresholding | OOF probability scan | Selects accuracy‑maximizing threshold (tie‑break strategies supported). |
| Metrics | Median + mean CV | Median reduces outlier fold volatility; mean retained for reference. |
| Performance | Demo mode | Small grids & fewer folds (≈ 2–3 min). |
| Full mode | Exhaustive expansion | Larger grids & adaptive growth (longer runtime, better ceiling). |
| Observability | Inference latency timing | Average per‑prediction latency computed after final fit. |
| Reproducibility | Deterministic seeds | Stable comparison across runs. |
| Tooling | CI pipeline | Lint → Tests → Demo run to guarantee health. |
| Extensibility | Modular design | Clear separation: search, threshold, evaluation, utils. |
| Packaging | Single bundle artifact | Stores pipeline, best params, threshold & summary metrics. |

---

## 🧠 Methodology Overview

### 1. Adaptive Median-Based Hyperparameter Expansion
1. Start with seed grids for each model (LogReg, RF, XGB, etc.).
2. Evaluate all combinations (CV).
3. Identify *best* parameter set by median CV accuracy (primary) then mean & std.
4. For each tunable dimension:
   - If best value sits at a boundary **and** median improvement over its nearest neighbor ≥ delta (e.g. 0.002),
   - Expand outward using scale‑aware rules (log, int stepping, paired rules for `learning_rate` ↔ `n_estimators`).
5. Repeat until no dimension justifies expansion or global improvement stagnates.

Median is less sensitive to single noisy folds than mean, producing more stable expansion signals.

### 2. OOF Threshold Optimization
1. Refit the best pipeline config per model.
2. Use cross-validated `predict_proba` to produce out‑of‑fold probabilities.
3. Evaluate accuracy over a candidate threshold grid (0.00 → 1.00).
4. Select threshold using:
   - Highest accuracy, and
   - Tie‑break by closeness to 0.50 (default) or configured rule.
5. Report delta from default 0.5 to highlight added value.

### 3. Final Model Bundle
Stores:
- `pipeline` (preprocessor + classifier)
- Sanitized best params
- Median & mean CV accuracies
- Chosen optimal threshold
- Threshold gain vs baseline
- Inference latency benchmark

---

## 🧩 Repository Structure

```
titanic-automl/
  data/
    data_raw/
      train.csv
      test.csv
  src/titanic_automl/
    __init__.py
    cli.py               # Entry point (demo/full modes)
    model_search.py      # Adaptive expansion & model orchestration
    threshold.py         # OOF probability & threshold search
    evaluation.py        # Latency + final summary
    utils.py             # Coercion, sorting, guards, normalization
  tests/
    test_utils.py
    test_threshold.py
    test_smoke_pipeline.py
  scripts/
    run_demo.sh
    run_full.sh
  notebooks/
    original_exploration.ipynb   # (Optional cleaned EDA / provenance)
  artifacts/          # (gitignored) model bundles / logs
  .github/workflows/ci.yml
  requirements.txt
  pyproject.toml
  README.md
  LICENSE (MIT)
  .gitignore
```

---

## ⚡ Quick Start

> Assumes you have `train.csv` and `test.csv` from the Kaggle Titanic dataset in the repo root.

```bash
git clone https://github.com/<your-user>/titanic-automl.git
cd titanic-automl
python -m venv .venv && source .venv/bin/activate  # (Linux/macOS)
pip install -r requirements.txt

# Fast demo (reduced search; ~2–3 mins)
python -m titanic_automl.cli --mode demo --data-dir data/data_raw --output-dir artifacts

# Full adaptive search (longer)
python -m titanic_automl.cli --mode full --data-dir data/data_raw --output-dir artifacts
```

Output sample (demo mode):
```
===== MODEL: RandomForest =====
[ITER 1] RandomForest: Evaluating 12 combos ...
[BEST] Iter 1 RandomForest: median=0.8520 mean=0.8462 std=0.0181
[STOP] No expansions triggered (iteration 1) ...
[THRESHOLD][RandomForest] Default@0.5=0.8429 | BestThr=0.520 Acc=0.8462 Gain=0.0033
...
Model: RandomForest
Median CV Accuracy: 0.8539
Selected Threshold: 0.520
Avg Inference Latency (sec/pred): 0.00078
```

---

## 🛠 Command Line Interface

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `demo` | `demo` (fast) or `full` (expanded) |
| `--data-dir` | `.` | Directory holding `train.csv`, `test.csv` |
| `--output-dir` | `artifacts` | Where artifacts (*.pkl) are saved |
| `--help` |  | Show usage |

---

## 📊 Example Metrics (Illustrative)

| Mode | Model (Best) | Median CV Acc | Mean CV Acc | OOF Threshold | OOF Gain | Latency (s/pred) | Runtime |
|------|--------------|---------------|-------------|---------------|----------|------------------|---------|
| Demo | XGBoost      | 0.860 – 0.865 | 0.845 – 0.850 | 0.50–0.52     | +0.0–0.01 | ~0.0005–0.0010   | ~2–3 m |
| Full | XGBoost/RF   | 0.865 – 0.870 | 0.848 – 0.856 | 0.50–0.66     | +0.0–0.02 | ~0.0005–0.0012   | 15–30 m |

> Replace with your actual recorded table after first full run.

---

## ⏱ Latency Measurement

We time repeated `predict` calls (`n` small batched in memory) to give an **approximate per‑sample inference latency**. This helps communicate deployability or compare model complexity decisions.

---

## 🧪 Testing & CI

The GitHub Actions workflow runs on every push / PR:

1. Install & cache dependencies
2. Lint (`ruff`, `black --check`)
3. Run unit tests (`pytest`)
4. Run smoke integration (demo mode)
5. (Optionally add coverage badge)

Run tests locally:

```bash
pytest -q
```

---

## 🔍 Logging & Interpretability (Future Hooks)

- Hook points exist to add:
  - Permutation importance
  - SHAP value computation
  - Calibration curves
  - Error slicing (e.g., by passenger class or gender)

---

## 🧱 Design Decisions

| Decision | Rationale |
|----------|-----------|
| Median CV accuracy | Reduces fold outlier bias vs mean. |
| Expansion threshold | Avoids grid explosion / noise chasing. |
| Paired expansion (learning rate ↔ n_estimators) | Counteracts interaction effect on ensemble quality. |
| OOF threshold tuning | Provides a robust probability-driven decision boundary, not reliant on training fold bias. |
| Hardening int/None coercion | Prevents silent failures when pandas casts `None → NaN`. |
| Demo/full separation | Ensures quick iteration + deeper reproducibility path. |

---

## 🧬 Roadmap

| Status | Idea | Notes |
|--------|------|-------|
| ⏳ | Add feature engineering module | Titles, family size, cabin deck, tickets grouping |
| ⏳ | Add ensembling (stacking / voting) | Combine top 2–3 models |
| ⏳ | Add calibration (Platt / isotonic) | Improve probability quality |
| ⏳ | Dockerfile + Make target | Repro in container |
| ⏳ | Coverage & Codecov badge | Quality signal |
| ⏳ | SHAP / interpretability report | Model transparency |
| ⏳ | HTML report artifact | One-click summary |
| ✅ | Adaptive grid expansion | Current |
| ✅ | OOF threshold selection | Current |
| ✅ | Latency benchmark | Current |

---

## 🗂 Data Notes

Download Titanic dataset (`train.csv`, `test.csv`) from [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic).  
Place in repository root or specify `--data-dir`.  
No raw data is versioned (respect size & license constraints).  

---

## 🧪 Reproducing Full Run (Long)

```bash
python -m titanic_automl.cli --mode full --data-dir . --output-dir artifacts \
  2>&1 | tee logs/full_run_$(date +%Y%m%d_%H%M).log
```

After completion inspect:
- `artifacts/best_model_with_threshold.pkl`
- Latency & threshold summary printed at tail of log.

---

## 📦 Artifact Structure

`best_model_with_threshold.pkl` contains:

| Key | Description |
|-----|-------------|
| `model_name` | Best model identifier |
| `pipeline` | sklearn Pipeline (preprocessor + estimator) |
| `best_params` | Sanitized classifier params |
| `median_accuracy` | CV median accuracy |
| `mean_accuracy` | CV mean accuracy |
| `threshold` | Selected decision threshold |
| `oof_default_acc` | Baseline 0.5 accuracy |
| `oof_best_acc` | Threshold-optimized accuracy |
| `oof_threshold_gain` | Improvement vs default |

Use programmatically:

```python
import joblib, pandas as pd
bundle = joblib.load("artifacts/best_model_with_threshold.pkl")
pipe = bundle["pipeline"]
pred = pipe.predict(pd.DataFrame({...}))
```

---

## 🛡 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for full text.

---

## ❤️ Acknowledgments

- [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic)
- scikit‑learn team
- XGBoost maintainers
- Open-source ecosystem contributors

---

## 🤝 Contributing

1. Fork & create feature branch: `git checkout -b feat/new-idea`
2. Run lint & tests: `make test` (if Makefile) or `pytest`
3. Submit PR with clear description + before/after runtime or accuracy if relevant

For ideas see **Roadmap** or open a **Discussion**.

---

## 🧭 Demo Narrative (Use in Interview)

1. Clone & install.
2. Run `demo` mode live → show adaptive expansion logs.
3. Display final metrics line (median, threshold, latency).
4. Open GitHub Actions page → green checks.
5. Briefly open `model_search.py` to highlight expansion criteria.
6. (Optional) Show threshold DataFrame or diff between 0.5 and optimized threshold.

---

## 📌 Badges (How to Add Later)

After adding coverage:
```
![Coverage](https://img.shields.io/badge/coverage-XX%25-brightgreen)
```

If adding Codecov:
```
[![codecov](https://codecov.io/gh/<user>/titanic-automl/branch/main/graph/badge.svg)](https://codecov.io/gh/<user>/titanic-automl)
```

---

## 📨 Support / Questions

Open an Issue for bugs, or a Discussion for conceptual questions / enhancements.

---

**Happy adaptive modeling!**  
If you use or extend this, a star ⭐ really helps visibility.
