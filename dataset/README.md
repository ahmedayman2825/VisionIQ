# 📂 dataset/

This folder holds the training data used to train the VisionIQ age/gender classifier. The dataset is **not tracked in git** — download it separately and place it here.

## Dataset

**FairFace** — a face image dataset balanced across race, gender, and age, used here for age-bracket and gender classification.

- Source: [Kaggle — FairFace](https://www.kaggle.com/datasets/aibloy/fairface)

## Expected structure

```
dataset/
└── FairFace/
    ├── train_labels.csv
    ├── val_labels.csv
    └── <image files>
```

`train_labels.csv` and `val_labels.csv` must each contain at minimum a `file`, `age`, and `gender` column, matching the format expected by `src/dataset.py`.

## Labels

**Age classes:** `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`

**Gender classes:** `Male`, `Female`

If your CSV uses a different age/gender label format, run `src/fix_fairface_labels.py` first to convert it to the expected format.
