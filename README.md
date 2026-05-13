# Two-Tower Uplift Model — Scooter User Retention

Code for a master's thesis. Implements a Two-Tower MLP uplift model that estimates the individual treatment effect of a gift (incentive) on the probability of a user returning to a scooter-sharing service.

## Repository structure

```
├── финал_diploma.ipynb          # main notebook with the full pipeline
├── two_tower_final_seed99.pth   # pre-trained model weights (seed=99)
├── requirements.txt             # Python dependencies
└── README.md
```

## Data

The data files are hosted on Yandex Disk (too large to include in the repository):

**[Download data from Yandex Disk](https://disk.yandex.ru/d/CCKixKyzMchHSA)**

Place both files in the same directory as the notebook before running:

| File | Description |
|------|-------------|
| `df_combined_noisy.csv` | AB-experiment data with injected noise |
| `user_features.csv` | Pre-experiment behavioural features per user |

## Notebook outline

1. **Data Loading** — load both sources, clean, create the target variable (at least one ride within 30 days after treatment), merge into a single dataframe
2. **Feature Preparation** — encode categorical features, scale numerics, build counterfactual treatment vectors
3. **Train / Val / Test Split** — stratified split by segment and treatment flag
4. **Two-Tower Architecture** — two independent MLP towers (user tower + treatment tower) concatenated before a shared head
5. **Metric Functions** — Qini AUC, Uplift AUC, Qini curve
6. **Visualisation Functions** — plotting utilities
7. **Model Training** — training with IPTW-weighted BCE loss and early stopping; loads pre-trained weights if the `.pth` file is present
8. **Baseline** — T-Learner with logistic regression
9. **Evaluation Metrics** — model comparison on the test set
10. **Qini Curves** — overall plot and per-segment breakdown (active_users / return_users)
11. **ATE** — average treatment effect by gift type and segment
12. **SHAP** — model explanations via GradientExplainer, PDP plots for key features

## Reproducing results

The model was trained with a fixed `SEED = 99`. If `two_tower_final_seed99.pth` is present, the notebook loads the saved weights and skips training — results will be identical.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run

```bash
jupyter notebook финал_diploma.ipynb
```

Make sure `df_combined_noisy.csv`, `user_features.csv`, and `two_tower_final_seed99.pth` are in the same directory as the notebook.
