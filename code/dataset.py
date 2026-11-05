"""CSV-backed data loading without rewriting existing splits."""
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent


def read_rows(csv_path, require_masks=False):
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    with open(csv_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'xrays', 'masks'}.issubset(reader.fieldnames or []):
            raise ValueError(f'{csv_path}: required columns are xrays,masks')
        rows = list(reader)
    if not rows:
        raise ValueError(f'Empty CSV: {csv_path}')
    seen = set()
    for row in rows:
        for key in ('xrays', 'masks'):
            value = (row[key] or '').strip()
            row[key] = None if value.lower() in ('', 'none', 'nan', 'null') else Path(value.replace('\\', '/'))
            if row[key] is not None:
                if not row[key].is_absolute():
                    row[key] = ROOT / row[key]
                row[key] = row[key].resolve()
                if not row[key].is_file():
                    raise FileNotFoundError(row[key])
        if row['xrays'] is None or (require_masks and row['masks'] is None):
            raise ValueError(f'Missing image or required mask: {row}')
        if row['xrays'] in seen:
            raise ValueError(f'Duplicate image in {csv_path}: {row["xrays"]}')
        seen.add(row['xrays'])
        if row['masks']:
            with Image.open(row['xrays']) as x, Image.open(row['masks']) as y:
                if x.size != y.size:
                    raise ValueError(f'Image/mask dimensions differ: {row}')
    return rows


def load_mask(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert('L'))
    return (a > (0 if a.max() <= 1 else 127)).astype(np.float32)


class KneeDataset(Dataset):
    def __init__(self, csv_path, image_size=256, augment=False, require_masks=True):
        self.rows = read_rows(csv_path, require_masks)
        self.size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row['xrays']) as im:
            x = im.convert('L').resize((self.size, self.size), Image.Resampling.BILINEAR)
        y = (Image.fromarray((load_mask(row['masks']) * 255).astype(np.uint8))
             if row['masks'] else Image.new('L', x.size))
        y = y.resize(x.size, Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < .5:
                x, y = ImageOps.mirror(x), ImageOps.mirror(y)
            angle = random.uniform(-10, 10)
            x = x.rotate(angle, resample=Image.Resampling.BILINEAR)
            y = y.rotate(angle, resample=Image.Resampling.NEAREST)
        return (torch.from_numpy(np.array(x, dtype=np.float32)[None] / 255),
                torch.from_numpy((np.array(y)[None] > 127).astype(np.float32)))
