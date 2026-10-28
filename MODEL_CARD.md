---
library_name: sklearn
license: mit
tags:
  - tabular-classification
  - sklearn
  - mnist1d
datasets:
  - prteek/mnist1d_1v7
metrics:
  - accuracy
---

# mnist1d_clf_1v7

A binary classifier that tells apart the digits **1** and **7** in the
[MNIST-1D](https://github.com/greydanus/mnist1d) dataset.

## Model

A scikit-learn pipeline: `StandardScaler` followed by `LogisticRegression(max_iter=1000)`.

## Data

Trained on [`prteek/mnist1d_f`](https://huggingface.co/datasets/prteek/mnist1d_f): the
`y == 1` and `y == 7` rows of the `christopher/mnist1d` train split (800 rows). Each
example has 40 features (`x_0` … `x_39`) and a label `y`.

80% of the rows were used for training and 20% were held out for testing, split
stratified on `y` with `random_state=42`.

## Results

| Metric   | Held-out test set (160 rows) |
|----------|------------------------------|
| Accuracy | 0.887                        |

## Usage

```python
import joblib
from huggingface_hub import hf_hub_download

model = joblib.load(hf_hub_download("prteek/mnist1d_clf", "model.joblib"))
preds = model.predict(X)  # X: DataFrame with columns x_0 … x_39; returns 1 or 7
```

## Limitations

- Only distinguishes 1 from 7; other digits get labelled as one of the two.
- Accuracy comes from a single 160-row test split, so it is a rough estimate.
- `joblib` files can run arbitrary code when loaded, so only load models you trust.
