PYTHONPATH := src
RAW_DIR := data/raw
PROCESSED_DIR := data/processed
ZIP_FILE := $(RAW_DIR)/h-and-m-personalized-fashion-recommendations.zip

.PHONY: help download-data remove-zip preprocess train-baseline train-baseline-databricks train-hybrid train-hybrid-databricks train-hybrid-fresh train-hybrid-fresh-databricks evaluate-final evaluate-final-databricks evaluate-final-fresh evaluate-final-fresh-databricks evaluate-cold-start evaluate-cold-start-databricks evaluate-policy-impact run-api run-ui test
help:
	@echo "Available targets:"
	@echo "  make download-data - Download + extract H&M Kaggle competition files"
	@echo "  make remove-zip    - Remove downloaded Kaggle zip file"
	@echo "  make preprocess    - Clean raw data and write parquet outputs"
	@echo "  make train-baseline - Train and evaluate popularity + SVD baselines"
	@echo "  make train-baseline-databricks - Train and track the run in Databricks MLflow"
	@echo "  make train-hybrid   - Train, tune, and evaluate content + hybrid models"
	@echo "  make train-hybrid-databricks - Track hybrid comparison in Databricks MLflow"
	@echo "  make train-hybrid-fresh - Train hybrid with fresh-item exposure enabled"
	@echo "  make train-hybrid-fresh-databricks - Track fresh-item hybrid run in Databricks"
	@echo "  make evaluate-final - Evaluate frozen models on the held-out test split"
	@echo "  make evaluate-final-databricks - Track final test metrics in Databricks"
	@echo "  make evaluate-final-fresh - Final held-out eval with fresh-item exposure"
	@echo "  make evaluate-final-fresh-databricks - Track fresh-item final eval in Databricks"
	@echo "  make evaluate-cold-start - Evaluate new-user fallback strategies"
	@echo "  make evaluate-cold-start-databricks - Track cold-start results in Databricks"
	@echo "  make evaluate-policy-impact - Simulate baseline vs hybrid policy lift"
	@echo "  make run-api       - Start the FastAPI recommendation server"
	@echo "  make run-ui        - Start the Streamlit product discovery experience"
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
	MLFLOW_TRACKING_URI= PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts

train-baseline-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts --tracking-uri databricks --experiment-name /Shared/reco-nova-baselines

train-hybrid:
	MLFLOW_TRACKING_URI= PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train_hybrid --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/hybrid

train-hybrid-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train_hybrid --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/hybrid --tracking-uri databricks --experiment-name /Shared/reco-nova-hybrid

train-hybrid-fresh:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train_hybrid --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/hybrid --include-fresh-catalog-items --min-fresh-in-top-k 1

train-hybrid-fresh-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.train_hybrid --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/hybrid --include-fresh-catalog-items --min-fresh-in-top-k 1 --tracking-uri databricks --experiment-name /Shared/reco-nova-hybrid

evaluate-final:
	MLFLOW_TRACKING_URI= PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_final --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/final --report-path docs/offline_evaluation_report.md

evaluate-final-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_final --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/final --report-path docs/offline_evaluation_report.md --tracking-uri databricks --experiment-name /Shared/reco-nova-final-evaluation

evaluate-final-fresh:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_final --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/final --report-path docs/offline_evaluation_report.md --include-fresh-catalog-items --min-fresh-in-top-k 1

evaluate-final-fresh-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_final --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/final --report-path docs/offline_evaluation_report.md --include-fresh-catalog-items --min-fresh-in-top-k 1 --tracking-uri databricks --experiment-name /Shared/reco-nova-final-evaluation

evaluate-cold-start:
	MLFLOW_TRACKING_URI= PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_cold_start --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/cold_start --report-path docs/cold_start_report.md

evaluate-cold-start-databricks:
	PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_cold_start --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/cold_start --report-path docs/cold_start_report.md --tracking-uri databricks --experiment-name /Shared/reco-nova-cold-start

evaluate-policy-impact:
	MLFLOW_TRACKING_URI= PYTHONPATH=$(PYTHONPATH) python -m reco_nova.evaluate_policy_impact --processed-dir $(PROCESSED_DIR) --artifacts-dir artifacts/policy_impact --report-path docs/policy_impact_report.md

run-api:
	PYTHONPATH=$(PYTHONPATH) uvicorn reco_nova.api:app --host 0.0.0.0 --port 8000

run-ui:
	PYTHONPATH=$(PYTHONPATH) streamlit run src/reco_nova/app.py --server.port 8501

test:
	PYTHONPATH=$(PYTHONPATH) python -m pytest -q