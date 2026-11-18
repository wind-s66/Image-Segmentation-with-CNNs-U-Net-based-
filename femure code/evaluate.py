
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arg import get_args
from dataset import KneeDataset
from model import UNet


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = Path(args.checkpoint)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    config = checkpoint["config"]

    model = UNet(config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print("Loaded model:", checkpoint_path)

    if args.mode == "evaluate":
        csv_path = args.val_csv
        require_masks = True
    else:
        csv_path = args.test_csv
        require_masks = False

    dataset = KneeDataset(
        csv_path,
        image_size=config["image_size"],
        require_masks=require_masks
    )

    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    dice_scores = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]

            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            logits = model(x)
            probability = torch.sigmoid(logits)
            mask = (probability >= args.threshold).float()

            prediction = mask[0, 0].cpu().numpy()

            Image.fromarray(
                (prediction * 255).astype(np.uint8)
            ).save(
                output_dir / f"prediction_{i + 1}.png"
            )

            if require_masks:
                intersection = (mask * y).sum()

                dice = (2 * intersection + 1e-7) / (
                    mask.sum() + y.sum() + 1e-7
                )

                score = dice.item()
                dice_scores.append(score)

                print(
                    f"Image {i + 1}: Dice Score = {score:.4f}"
                )
            else:
                print(f"Image {i + 1}: Prediction saved")

    if require_masks and dice_scores:
        mean_dice = sum(dice_scores) / len(dice_scores)

        print(f"\nMean Dice Score: {mean_dice:.4f}")
        print(f"Mean Dice Score: {mean_dice * 100:.2f}%")

        with open(output_dir / "dice_score.txt", "w") as f:
            for i, score in enumerate(dice_scores):
                f.write(f"Image {i + 1}: {score:.4f}\n")

            f.write(f"\nMean Dice Score: {mean_dice:.4f}\n")

    print("Results saved to:", output_dir)


if __name__ == "__main__":
    args = get_args()

    if args.mode == "train":
        args.mode = "evaluate"

    evaluate(args)
