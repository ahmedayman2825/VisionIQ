import pandas as pd

TRAIN_CSV = "dataset/FairFace/train_labels.csv"
VAL_CSV = "dataset/FairFace/val_labels.csv"

AGE_MAP = {
    "9-Mar": "3-9",
    "19-Oct": "10-19",
    "69-Jun": "60-69",
    "more than 70": "70+"
}


def fix_labels(csv_path):

    df = pd.read_csv(csv_path)

    df["age"] = (
        df["age"]
        .astype(str)
        .str.strip()
        .replace(AGE_MAP)
    )

    df.to_csv(csv_path, index=False)

    print(f"✓ Fixed: {csv_path}")


fix_labels(TRAIN_CSV)
fix_labels(VAL_CSV)

print("\nDone!")