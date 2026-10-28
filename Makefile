.PHONY: process_data train
all: process_data train update_model_card

DATASET_REPO=mnist1d_1v7
MODEL_REPO=mnist1d_clf_1v7

process_data:
	hf jobs uv run --name process-mnist1d --secrets HF_TOKEN process.py --repo-id prteek/$(DATASET_REPO)

train:
	hf jobs uv run --name train-mnist1d --secrets HF_TOKEN train.py --dataset-repo-id prteek/$(DATASET_REPO) --model-repo-id prteek/$(MODEL_REPO)

update_model_card:
	hf repos cp MODEL_CARD.md hf://prteek/$(MODEL_REPO)/README.md
