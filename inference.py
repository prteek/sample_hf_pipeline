# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb>=1.4.0",
#     "huggingface-hub>=2.0.0",
#     "requests>=2.32.0",
# ]
# ///
import duckdb
import requests
from huggingface_hub import get_inference_endpoint, get_token

ENDPOINT = "mnist1d-clf-1v7"
DATASET_REPO_ID = "prteek/mnist1d_1v7"
N_SAMPLES = 10

X = duckdb.sql(f"""
    SELECT * EXCLUDE (y)
    FROM 'hf://datasets/{DATASET_REPO_ID}/train.parquet'
    USING SAMPLE {N_SAMPLES} ROWS
""").fetchall()

endpoint = get_inference_endpoint(ENDPOINT)
# Generous timeout: a scaled-to-zero endpoint cold starts on the first request
response = requests.post(
    endpoint.url,
    headers={"Authorization": f"Bearer {get_token()}"},
    json={"inputs": X},
    timeout=300,
)
response.raise_for_status()

print(response.json()["predictions"])
