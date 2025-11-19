import os
from pathlib import Path

import kagglehub

# Ensure downloads/cache go into the local `Dataset` folder
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_CACHE = PROJECT_ROOT / "Dataset"
os.environ.setdefault("KAGGLEHUB_CACHE", str(DATASET_CACHE))
DATASET_CACHE.mkdir(parents=True, exist_ok=True)

# Try to download; if credentials are missing or network fails, show intended path instead
try:
	# Download latest version (this will use KAGGLEHUB_CACHE as the base cache folder)
	path = kagglehub.dataset_download("joykaihatu/image-caption-indonesia")
	print("Path to dataset files:", path)
except Exception as e:
	print("Download failed or requires Kaggle credentials.")
	print("Intended cache folder:", os.environ.get("KAGGLEHUB_CACHE"))
	print("Error:", e)