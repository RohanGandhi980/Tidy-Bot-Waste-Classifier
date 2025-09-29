import os, random, shutil, requests, tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR, VAL_DIR = DATA_DIR / "train", DATA_DIR / "val"

TACO_DIR = DATA_DIR / "archive (3)" / "data"

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ZIP = DATA_DIR / "coco_val2017.zip"
COCO_DIR = DATA_DIR / "coco"

def download_file(url, dest):
    if dest.exists():
        print(f"✔ {dest.name} already exists, skipping download.")
        return
    print(f"⬇️ Downloading {url}")
    resp = requests.get(url, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"Download failed with status {resp.status_code}")
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in resp.iter_content(1024 * 1024):
            if not chunk:
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

def main():
    prepare_folders()
    taco_imgs = []
    if TACO_DIR.exists():
        for batch in TACO_DIR.glob("batch_*"):
            taco_imgs.extend(list(batch.glob("*.jpg")))

    if not taco_imgs:
        raise RuntimeError(f"⚠️ No TACO images found in {TACO_DIR}, check path.")

    random.shuffle(taco_imgs)
    split = int(0.8 * len(taco_imgs))

    for i, p in enumerate(taco_imgs):
        unique_name = f"{p.parent.stem}_{p.name}"
        dest = TRAIN_DIR / "trash" / unique_name if i < split else VAL_DIR / "trash" / unique_name
        shutil.copy(p, dest)
    print(f"✔ TACO: {len(taco_imgs)} trash images copied safely")

    # ===== COCO =====
    download_file(COCO_URL, COCO_ZIP)
    if not (COCO_DIR / "val2017").exists():
        unzip_file(COCO_ZIP, COCO_DIR)

    coco_imgs = list((COCO_DIR / "val2017").glob("*.jpg"))
    random.shuffle(coco_imgs)

    sampled = coco_imgs[:len(taco_imgs)]  # balance with TACO
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
