from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import KneeDataset
from model import UNet


def evaluate(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )

    config = checkpoint["config"]

    model = UNet(config["base_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    
    if args.mode == "evaluate":
        csv_path = args.val_csv
    else:
        csv_path = args.test_csv

    dataset = KneeDataset(
        csv_path,
        image_size=config["image_size"],
        require_masks=(args.mode == "evaluate")
    )

    
    output_dir = Path(args.output_dir) / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    dice_scores = []

   
    with torch.no_grad():

        for i in range(len(dataset)):

            
            x, y = dataset[i]

            x = x.unsqueeze(0).to(device)

            
            logits = model(x)

            
            probability = torch.sigmoid(logits)

            
            mask = (probability >= args.threshold).float()

            
            prediction = mask[0, 0].cpu().numpy()

            prediction_image = Image.fromarray(
                (prediction * 255).astype(np.uint8)
            )

            prediction_image.save(
                output_dir / f"prediction_{i + 1}.png"
            )

      
            if args.mode == "evaluate":

                y = y.unsqueeze(0).to(device)

                intersection = (mask * y).sum()

                dice = (
                    2 * intersection + 1e-7
                ) / (
                    mask.sum() + y.sum() + 1e-7
                )

                dice_scores.append(dice.item())
                print(f"Image {i + 1}: Dice Score = {dice.item():.4f}")


    if args.mode == "evaluate":

        mean_dice = sum(dice_scores) / len(dice_scores)

        print("Mean Dice:", mean_dice)

    else:
        print("Prediction completed!")

    print("Results saved to:", output_dir)


if __name__ == "__main__":
    from arg import get_args

    args = get_args()

    if args.mode == "train":
        args.mode = "evaluate"

    evaluate(args)