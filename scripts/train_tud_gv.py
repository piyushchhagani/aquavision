"""
AquaVision - Automated Dataset Downloader & YOLO Trainer for TUD-GV Floating Litter
"""

import os
import shutil
import zipfile
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "tud_gv"
PROCESSED_DIR = DATA_DIR / "processed" / "tud_gv"
MODELS_DIR = PROJECT_ROOT / "models"

ZENODO_FILES = {
    "images.zip": "https://zenodo.org/records/13730228/files/images.zip?download=1",
    "labels_txt.zip": "https://zenodo.org/records/13730228/files/labels_txt.zip?download=1",
    "classes.txt": "https://zenodo.org/records/13730228/files/classes.txt?download=1",
}


def download_file(url: str, dest: Path):
    if not dest.exists():
        print(f"Downloading {dest.name}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Downloaded {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"{dest.name} already exists. Skipping download.")


def prepare_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, url in ZENODO_FILES.items():
        download_file(url, RAW_DIR / fname)

    # Extract
    extracted_images = RAW_DIR / "images"
    extracted_labels = RAW_DIR / "labels"
    if not extracted_images.exists():
        print("Extracting images.zip...")
        with zipfile.ZipFile(RAW_DIR / "images.zip", "r") as z:
            z.extractall(RAW_DIR)

    if not extracted_labels.exists():
        print("Extracting labels_txt.zip...")
        with zipfile.ZipFile(RAW_DIR / "labels_txt.zip", "r") as z:
            z.extractall(RAW_DIR)

    # Setup train/val/test splits (1050 train, 225 val, 226 test)
    image_files = sorted(list(RAW_DIR.glob("**/*.jpg")) + list(RAW_DIR.glob("**/*.png")))
    valid_pairs = []
    for img in image_files:
        lbl = RAW_DIR / "labels_txt" / f"{img.stem}.txt"
        if not lbl.exists():
            lbl = RAW_DIR / f"{img.stem}.txt"
        if lbl.exists():
            valid_pairs.append((img, lbl))

    print(f"Found {len(valid_pairs)} matched image-annotation pairs.")
    train_pairs, test_pairs = train_test_split(valid_pairs, test_size=226, random_state=42)
    train_pairs, val_pairs = train_test_split(train_pairs, test_size=225, random_state=42)

    for split_name, pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
        img_dest = PROCESSED_DIR / "images" / split_name
        lbl_dest = PROCESSED_DIR / "labels" / split_name
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)

        for img, lbl in pairs:
            shutil.copy2(img, img_dest / img.name)
            shutil.copy2(lbl, lbl_dest / lbl.name)

    # Generate data.yaml with absolute paths
    yaml_path = PROCESSED_DIR / "tud_gv.yaml"
    yaml_content = f"""path: {PROCESSED_DIR.as_posix()}
train: images/train
val: images/val
test: images/test
names:
  0: litter
"""
    yaml_path.write_text(yaml_content)
    print(f"Dataset prepared successfully at: {PROCESSED_DIR}")
    return yaml_path


def train_model(data_yaml: Path, epochs=30, model_size="yolo11n.pt"):
    print(f"Initializing training with {model_size} for {epochs} epochs...")
    model = YOLO(model_size)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        save=True,
        project=str(PROJECT_ROOT / "outputs" / "train"),
        name="tud_gv_run",
    )

    best_weights = PROJECT_ROOT / "outputs" / "train" / "tud_gv_run" / "weights" / "best.pt"
    if best_weights.exists():
        dest = MODELS_DIR / "tud_gv_best.pt"
        shutil.copy2(best_weights, dest)
        print(f"SUCCESS: Best trained model copied to: {dest}")
    return results


if __name__ == "__main__":
    yaml_cfg = prepare_dataset()
    train_model(yaml_cfg, epochs=30, model_size="yolo11n.pt")
