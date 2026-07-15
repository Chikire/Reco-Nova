PYTHONPATH := src
RAW_DIR := data/raw
PROCESSED_DIR := data/processed
ZIP_FILE := $(RAW_DIR)/h-and-m-personalized-fashion-recommendations.zip

.PHONY: help download-data remove-zip preprocess train-baseline train-baseline-databricks train-hybrid train-hybrid-databricks test
help:
	@echo "Available targets:"
	@echo "  make download-data - Download + extract H&M Kaggle competition files"
	@echo "  make remove-zip    - Remove downloaded Kaggle zip file"
	@echo "  make preprocess    - Clean raw data and write parquet outputs"
	@echo "  make train-baseline - Train and evaluate popularity + SVD baselines"
	@echo "  make train-baseline-databricks - Train and track the run in Databricks MLflow"
	@echo "  make train-hybrid   - Train, tune, and evaluate content + hybrid models"
	@echo "  make train-hybrid-databricks - Track hybrid comparison in Databricks MLflow"
	@echo "  make test          - Run the automated test suite"
	
download-data:
	mkdir -p $(RAW_DIR)
	bash scripts/download_data.sh -p $(RAW_DIR)
	python -m zipfile -e $(ZIP_FILE) $(RAW_DIR)

remove-zip:
	rm -f $(ZIP_FILE)

preprocess:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.preprocess --raw-dir $(RAW_DIR) --processed-dir $(PROCESSED_DIR)

train-baseline:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts

train-baseline-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts --tracking-uri databricks --experiment-name /Shared/reco-nova-baselines

train-hybrid:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train_hybrid --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/hybrid

train-hybrid-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train_hybrid --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/hybrid --tracking-uri databricks --experiment-name /Shared/reco-nova-hybrid

test:
	PYTHONPATH=$(PYTHONPATH) python -m pytest -q
