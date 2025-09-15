import os, shutil, random

dataset_dir = "dataset-resized"
output_dir = "dataset-split"

categories = os.listdir(dataset_dir)
split_ratio = [0.7, 0.15, 0.15]  # train, val, test

for category in categories:
    files = os.listdir(os.path.join(dataset_dir, category))
    random.shuffle(files)
    
    train_end = int(split_ratio[0] * len(files))
    val_end = int((split_ratio[0] + split_ratio[1]) * len(files))
    
    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }
    
    for split, split_files in splits.items():
        split_path = os.path.join(output_dir, split, category)
        os.makedirs(split_path, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(dataset_dir, category, f),
                        os.path.join(split_path, f))
