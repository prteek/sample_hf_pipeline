.PHONY: process_data train update_model_card upload_handler deploy inference delete_endpoint
all: process_data train update_model_card deploy

DATASET_REPO=mnist1d_1v7
MODEL_REPO=mnist1d_clf_1v7
ENDPOINT=mnist1d-clf-1v7

process_data:
	hf jobs uv run --name process-mnist1d --secrets HF_TOKEN process.py --repo-id prteek/$(DATASET_REPO)

train:
	hf jobs uv run --name train-mnist1d --secrets HF_TOKEN train.py --dataset-repo-id prteek/$(DATASET_REPO) --model-repo-id prteek/$(MODEL_REPO)

update_model_card:
	hf repos cp MODEL_CARD.md hf://prteek/$(MODEL_REPO)/README.md

# sklearn has no built-in serving framework, so ship a custom handler with the model
upload_handler:
	hf upload prteek/$(MODEL_REPO) deploy . --commit-message "Add inference handler"

# Smallest CPU instance, scales to zero after 15 min idle
deploy: upload_handler
	hf endpoints deploy $(ENDPOINT) --repo prteek/$(MODEL_REPO) --secrets HF_TOKEN \
		--framework pytorch --task custom \
		--accelerator cpu --instance-type intel-spr --instance-size x1 \
		--vendor aws --region eu-west-1 \
		--min-replica 0 --max-replica 1 --scale-to-zero-timeout 15

inference:
	uv run inference.py

delete_endpoint:
	hf endpoints delete $(ENDPOINT) --yes
