# split_val.py
import os, random, shutil

random.seed(42)
base = "data"

for cls in ["NORMAL", "PNEUMONIA"]:
    train_dir = os.path.join(base, "train", cls)
    val_dir = os.path.join(base, "val", cls)
    os.makedirs(val_dir, exist_ok=True)

    files = os.listdir(train_dir)
    n_move = int(len(files) * 0.15)
    for f in random.sample(files, n_move):
        shutil.move(os.path.join(train_dir, f), os.path.join(val_dir, f))
    print(f"{cls}: moved {n_move} images to val")