# prepare_dataset.py
import os, random, zipfile, shutil, requests
from pathlib import Path
from tqdm import tqdm

# =============================
# Config
# =============================
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR, VAL_DIR = DATA_DIR / "train", DATA_DIR / "val"

# TrashNet dataset (already downloaded & extracted)
TRASHNET_DIR = DATA_DIR / "trashnet" / "dataset-resized"

# COCO dataset (non-trash)
COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ZIP = DATA_DIR / "coco_val2017.zip"

COCO_DIR = DATA_DIR / "coco"

# =============================
# Utils
# =============================
def download_file(url, dest, max_mb=None):
    """Download file from URL if not exists (optional size cap)."""
    if dest.exists():
        print(f"✔ {dest.name} already exists, skipping download.")
        return

    print(f"⬇️ Downloading {url}")
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed with status {resp.status_code}")

    total = int(resp.headers.get("content-length", 0))
    cap = max_mb * 1024 * 1024 if max_mb else total

    with open(dest, "wb") as f, tqdm(total=cap, unit="B", unit_scale=True, desc=dest.name) as bar:
        size = 0
        for chunk in resp.iter_content(1024 * 1024):
            if not chunk:
                break
            size += len(chunk)
            if size > cap:
                break
            f.write(chunk)
            bar.update(len(chunk))
    print(f"✔ Downloaded {dest} ({dest.stat().st_size/1024/1024:.1f} MB)")


def unzip_file(src, dest):
    print("📦 Extracting COCO...")
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dest)


def prepare_folders():
    for split in ["train", "val"]:
        for cls in ["trash", "non_trash"]:
            (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)


def count_images():
    print("\n📊 Dataset summary:")
    for split in ["train", "val"]:
        for cls in ["trash", "non_trash"]:
            path = DATA_DIR / split / cls
            count = len(list(path.glob("*.jpg")))
            print(f"{split}/{cls}: {count} images")

# =============================
# Main
# =============================
def main():
    prepare_folders()

    # ===== Step 1: TrashNet (trash images) =====
    trash_imgs = []
    if TRASHNET_DIR.exists():
        for cls in TRASHNET_DIR.iterdir():
            if cls.is_dir():  # each class folder
                trash_imgs.extend(list(cls.glob("*.jpg")))

    if not trash_imgs:
        raise RuntimeError(f"⚠️ No TrashNet images found in {TRASHNET_DIR}")

    random.shuffle(trash_imgs)
    split = int(0.8 * len(trash_imgs))

    for i, p in enumerate(trash_imgs):
        unique_name = f"trashnet_{p.parent.stem}_{p.name}"
        dest = TRAIN_DIR / "trash" / unique_name if i < split else VAL_DIR / "trash" / unique_name
        shutil.copy(p, dest)

    print(f"✔ TrashNet: {len(trash_imgs)} trash images copied safely")

    # ===== Step 2: COCO (non-trash images) =====
    download_file(COCO_URL, COCO_ZIP, max_mb=3500)  # cap ~3.5GB
    if not (COCO_DIR / "train2017").exists():
        unzip_file(COCO_ZIP, COCO_DIR)

    coco_imgs = list((COCO_DIR / "val2017").glob("*.jpg"))
    random.shuffle(coco_imgs)

    # Balance size with TrashNet (~same count)
    sampled = coco_imgs[:len(trash_imgs)]
    split = int(0.8 * len(sampled))

    for i, p in enumerate(sampled):
        unique_name = f"coco_{p.name}"
        dest = TRAIN_DIR / "non_trash" / unique_name if i < split else VAL_DIR / "non_trash" / unique_name
        shutil.copy(p, dest)

    print(f"✔ COCO: {len(sampled)} non-trash images copied safely")

    count_images()
    print("\n✅ Balanced dataset ready at ./data")


if __name__ == "__main__":
    main()
