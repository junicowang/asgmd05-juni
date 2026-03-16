from pathlib import Path
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from utils_folder.helper import build_pipeline

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data" / "train.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
PIPELINE_FILE = ARTIFACT_DIR / "pipeline.pkl"
RANDOM_STATE = 42


def main():
    # 1. Load Data
    print("=" * 50)
    print("Step 1: Load Data")
    print("=" * 50)
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape}")

    X = df.drop(columns=['Transported'])
    y = df['Transported'].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # 2. Build & Evaluate Pipeline
    print("\n" + "=" * 50)
    print("Step 2: Build & Evaluate Pipeline")
    print("=" * 50)
    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(f"\n{classification_report(y_val, y_pred)}")

    # 3. Train on Full Data & Save
    print("=" * 50)
    print("Step 3: Train Full Data & Save")
    print("=" * 50)
    pipeline.fit(X, y)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_FILE, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Pipeline saved → {PIPELINE_FILE}")


if __name__ == "__main__":
    main()