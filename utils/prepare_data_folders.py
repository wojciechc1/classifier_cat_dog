import os
import random
import shutil
from pathlib import Path

def prepare_data_folders(source_cat, source_dog, target_dir):
    os.makedirs(target_dir, exist_ok=True)

    # Podzial
    split_ratio = 0.8

    def split_and_copy(class_name, source_path):
        images = list(Path(source_path).glob("*.jpg"))
        random.shuffle(images)

        split = int(len(images) * split_ratio)
        train_imgs = images[:split]
        val_imgs = images[split:]

        for split_name, imgs in zip(["train", "val"], [train_imgs, val_imgs]):
            out_dir = Path(target_dir) / split_name / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, out_dir / img.name)

    split_and_copy("cat", source_cat)
    split_and_copy("dog", source_dog)

    print("Dane przygotowane w folderze:", target_dir)
