# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=5.1.0",
#     "pandas>=3.1.0",
#     "pyarrow>=26.0.0",
# ]
# ///
import argparse

from datasets import load_dataset
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--repo-id", required=True)
args = parser.parse_args()

train = load_dataset("christopher/mnist1d", split="train")

train.set_format("numpy")
pdf = train.to_pandas()
df = pd.DataFrame(pdf["x"].tolist()).add_prefix("x_").assign(y=pdf["y"])

df_filtered = df.query("y==1 or y==7")

df_filtered.to_parquet(f"hf://datasets/{args.repo_id}/train.parquet", index=False)
print(f"Wrote {len(df_filtered)} rows to {args.repo_id}")
