PYTHONPATH := src
RAW_DIR := data/raw
ZIP_FILE := $(RAW_DIR)/h-and-m-personalized-fashion-recommendations.zip

.PHONY: help download-data remove-zip
help:
	@echo "Available targets:"
	@echo "  make download-data - Download + extract H&M Kaggle competition files"
	@echo "  make remove-zip    - Remove downloaded Kaggle zip file"
	
download-data:
	mkdir -p $(RAW_DIR)
	bash scripts/download_data.sh -p $(RAW_DIR)
	python -m zipfile -e $(ZIP_FILE) $(RAW_DIR)

remove-zip:
	rm -f $(ZIP_FILE)

