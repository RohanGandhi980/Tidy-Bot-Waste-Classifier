import os, argparse, json, glob, csv, random
import numpy as np
from PIL import Image
import tensorflow as tf

IMG_SIZE = 224
MODEL_DIR = "models"
DEFAULT_MODEL = os.path.join(MODEL_DIR, "tidybot_best.h5")
CLASS_MAP_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# ---------- loaders ----------
def load_model(model_path):
    print(f"[INFO] Loading model: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_class_map(path):
    with open(path) as f:
        cls = json.load(f)
    return {v: k for k, v in cls.items()}  # id -> name

# ---------- preprocessing / predict ----------
def preprocess(img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(x, 0)

def predict_one(model, id2name, img_path, topk=3):
    x = preprocess(img_path)
    probs = model.predict(x, verbose=0)[0]
    idx = probs.argsort()[::-1][:topk]
    return [(id2name[i], float(probs[i])) for i in idx]

# ---------- helpers ----------
def pick_random_image(root="dataset-split/test", klass=None):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    pool = []
    if klass:
        for e in exts:
            pool += glob.glob(os.path.join(root, klass, e))
    else:
        # all subfolders under test/
        for c in next(os.walk(root))[1]:
            for e in exts:
                pool += glob.glob(os.path.join(root, c, e))
    if not pool:
        raise FileNotFoundError(f"No images found under {root}" + (f"/{klass}" if klass else ""))
    return random.choice(pool)

# ---------- runners ----------
def run_single(args, model, id2name):
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    top = predict_one(model, id2name, args.image, args.topk)
    print(f"\nImage: {args.image}")
    for i, (label, conf) in enumerate(top, 1):
        print(f"{i}. {label}: {conf:.4f}")

def run_dir(args, model, id2name):
    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.webp")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(args.dir, e))
    files = sorted(files)
    if not files:
        print("No images found in directory."); return

    out_csv = args.out if args.out else os.path.join(MODEL_DIR, "predictions.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename","pred_label","pred_conf"] \
               + [f"top{i}_label" for i in range(2, args.topk+1)] \
               + [f"top{i}_conf" for i in range(2, args.topk+1)]
        writer.writerow(header)
        for fp in files:
            top = predict_one(model, id2name, fp, args.topk)
            pred_label, pred_conf = top[0]
            extra_labels = [lbl for lbl,_ in top[1:]]
            extra_confs  = [f"{conf:.6f}" for _,conf in top[1:]]
            row = [fp, pred_label, f"{pred_conf:.6f}"] + extra_labels + extra_confs
            writer.writerow(row)
    print(f"[INFO] Wrote predictions to {out_csv}")

# ---------- cli ----------
def parse_args():
    ap = argparse.ArgumentParser(description="Predict with TidyBot classifier")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--image", help="Path to a single image")
    g.add_argument("--dir",   help="Directory of images")
    ap.add_argument("--random", action="store_true",
                    help="Pick a random image from dataset-split/test")
    ap.add_argument("--sample", help="Pick a random image from this class (e.g., glass)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Path to .h5 model")
    ap.add_argument("--classes", default=CLASS_MAP_PATH, help="Path to class_indices.json")
    ap.add_argument("--topk", type=int, default=3, help="Top-K results to show")
    ap.add_argument("--out", help="(dir mode) CSV output path")
    return ap.parse_args()

# ---------- main ----------
if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model)
    id2name = load_class_map(args.classes)

    # Convenience: auto-pick an image if requested
    if args.random or args.sample:
        chosen = pick_random_image(klass=args.sample)
        print(f"[INFO] Auto-picked: {chosen}")
        args.image = chosen

    # Default behavior: single image if provided, else dir mode
    if args.image:
        run_single(args, model, id2name)
    elif args.dir:
        run_dir(args, model, id2name)
    else:
        # If nothing specified, pick one random test image
        chosen = pick_random_image()
        print(f"[INFO] Auto-picked (fallback): {chosen}")
        args.image = chosen
        run_single(args, model, id2name)
