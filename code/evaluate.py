"""Evaluate at original image resolution, or predict unlabeled test images."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from dataset import read_rows, load_mask
from model import UNet
from utils import get_device, seed_everything, write_csv, write_json


def evaluate(args):
    seed_everything(args.seed, args.threads)
    device = get_device(args.device)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f'Model not found: {args.checkpoint}. Train the model first.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = checkpoint['config']
    model = UNet(config['base_channels']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    csv_path = args.val_csv if args.mode == 'evaluate' else args.test_csv
    rows = read_rows(csv_path, require_masks=args.mode == 'evaluate')
    split_name = 'Validation' if args.mode == 'evaluate' else 'Test'
    print(f'{split_name}: {len(rows)} images; device={device}; '
          f'checkpoint epoch={checkpoint["epoch"]}', flush=True)
    out = Path(args.output_dir) / args.mode
    for folder in ('masks', 'overlays', 'comparisons'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    names = [r['xrays'].stem for r in rows]
    if len(set(names)) != len(names):
        raise ValueError('Output filenames would collide: duplicate image stems.')
    scores = []
    with torch.inference_mode():
        for index, row in enumerate(rows, start=1):
            with Image.open(row['xrays']) as im:
                original = im.convert('L')
            resized = original.resize((config['image_size'], config['image_size']), Image.Resampling.BILINEAR)
            x = torch.from_numpy(np.array(resized, dtype=np.float32)[None, None] / 255).to(device)
            probability = model(x).sigmoid()
            probability = F.interpolate(probability, size=(original.height, original.width),
                                        mode='bilinear', align_corners=False)
            prediction = probability[0, 0].cpu().numpy() >= args.threshold
            mask = Image.fromarray(prediction.astype(np.uint8) * 255)
            name = row['xrays'].stem + '.png'
            mask.save(out / 'masks' / name)
            rgb = np.array(original.convert('RGB'))
            overlay = rgb.copy()
            overlay[prediction] = (.55 * overlay[prediction] + .45 * np.array([255, 40, 40])).astype(np.uint8)
            Image.fromarray(overlay).save(out / 'overlays' / name)
            panels = [('X-ray', original.convert('RGB'))]
            score = None
            if row['masks']:
                truth = load_mask(row['masks']).astype(bool)
                total = prediction.sum() + truth.sum()
                score = float(2 * np.logical_and(prediction, truth).sum() / total) if total else 1.
                panels.append(('Ground truth', Image.fromarray(truth.astype(np.uint8)*255).convert('RGB')))
            panels.extend([('Prediction', mask.convert('RGB')), ('Overlay', Image.fromarray(overlay))])
            header = 24
            canvas = Image.new('RGB', (original.width * len(panels), original.height + header), 'white')
            draw = ImageDraw.Draw(canvas)
            for i, (title, panel) in enumerate(panels):
                canvas.paste(panel, (i * original.width, header))
                draw.text((i * original.width + 6, 5), title, fill='black')
            canvas.save(out / 'comparisons' / name)
            scores.append({'image': row['xrays'].name, 'dice': score})
            if index % 10 == 0 or index == len(rows):
                print(f'{split_name} progress: {index}/{len(rows)}', flush=True)
    labeled = [s['dice'] for s in scores if s['dice'] is not None]
    report = dict(split=str(csv_path), checkpoint=str(args.checkpoint),
                  checkpoint_epoch=checkpoint['epoch'], threshold=args.threshold,
                  resolution='original', images=len(rows), labeled_images=len(labeled),
                  mean_dice=float(np.mean(labeled)) if labeled else None,
                  std_dice=float(np.std(labeled)) if labeled else None,
                  note='Mean per-image foreground hard Dice; both empty = 1. '
                       'Unlabeled images have no Dice score.')
    write_csv(out / 'per_image_dice.csv', scores)
    write_json(out / 'metrics.json', report)
    print(f'Saved {len(rows)} predictions to {out}')
    if labeled:
        print(f'{split_name} Hard Dice: {report["mean_dice"]:.4f} '
              f'({len(labeled)}/{len(rows)} labeled images)')
    else:
        print(f'{split_name} prediction completed for all {len(rows)} images. '
              'Dice: N/A (no ground-truth masks).')


if __name__ == '__main__':
    from arg import get_args
    args = get_args()
    if args.mode == 'train':
        args.mode = 'evaluate'
    evaluate(args)
