# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=2.0.0",
#     "pandas>=3.1.0",
#     "pyarrow>=26.0.0",
#     "scikit-learn>=1.9.1",
# ]
# ///
import argparse

import joblib
import pandas as pd
from huggingface_hub import HfApi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-repo-id", required=True)
parser.add_argument("--model-repo-id", required=True)
args = parser.parse_args()

df = pd.read_parquet(f"hf://datasets/{args.dataset_repo_id}/train.parquet")
X, y = df.drop(columns="y"), df["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

joblib.dump(model, "model.joblib")

api = HfApi()
api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=args.model_repo_id,
    commit_message=f"Train logistic regression (test accuracy {accuracy:.3f})",
)
print(f"Uploaded model to {args.model_repo_id}")
