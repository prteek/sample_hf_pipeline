# sample_hf_pipeline

A minimal pipeline that runs on Hugging Face Jobs to train a binary classifier separating digits 1 and 7 in [MNIST-1D](https://github.com/greydanus/mnist1d).

- `process.py` builds the dataset and pushes it to `prteek/mnist1d_1v7`
- `train.py` trains the model and pushes it to `prteek/mnist1d_clf_1v7`
- `MODEL_CARD.md` is uploaded as the model repo's README

## Usage

```sh
make all
```

Requires the `hf` CLI logged in with an `HF_TOKEN`.
