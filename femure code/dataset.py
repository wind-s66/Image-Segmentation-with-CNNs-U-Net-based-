
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def get_path(path):
    path = str(path).strip().replace("\\", "/")
    if path.startswith("data/"):
        path = path[5:]

    return ROOT / path


class KneeDataset(Dataset):

    def __init__(self, csv_path, image_size=256,
                 augment=False, require_masks=True):

        self.size = image_size
        self.augment = augment
        self.require_masks = require_masks

        csv_path = get_path(csv_path)

        with open(csv_path, encoding="utf-8-sig") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):

        row = self.rows[index]

    
        x_path = get_path(row["xrays"])

        with Image.open(x_path) as im:
            x = im.convert("L")

        x = x.resize(
            (self.size, self.size),
            Image.Resampling.BILINEAR
        )

        
        mask_path = (row["masks"] or "").strip()

        if mask_path.lower() not in ("", "none", "nan", "null"):

            with Image.open(get_path(mask_path)) as im:
                y = im.convert("L")

            y = y.resize(
                (self.size, self.size),
                Image.Resampling.NEAREST
            )

        else:
            if self.require_masks:
                raise ValueError("Missing mask")

            y = Image.new("L", (self.size, self.size))

       
        if self.augment:

            if random.random() < 0.5:
                x = ImageOps.mirror(x)
                y = ImageOps.mirror(y)

            angle = random.uniform(-10, 10)

            x = x.rotate(
                angle,
                resample=Image.Resampling.BILINEAR
            )

            y = y.rotate(
                angle,
                resample=Image.Resampling.NEAREST
            )

     
        x = np.array(x, dtype=np.float32) / 255.0

        y = np.array(y, dtype=np.float32)
        threshold = 0 if y.max() <= 1 else 127
        y = (y > threshold).astype(np.float32)

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        return x, y
