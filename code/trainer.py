from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import KneeDataset
from model import UNet
from utils import segmentation_loss, dice_per_image, plot_history, write_csv


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_set = KneeDataset(args.train_csv, args.image_size, augment=True)
    val_set = KneeDataset(args.val_csv, args.image_size)

    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.bs,
        shuffle=False
    )

    model = UNet(args.base_channels).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )


    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_dice = 0
    history = []


    for epoch in range(args.epochs):

        model.train()
        train_loss = 0
        train_dice = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = segmentation_loss(logits, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(x)

            mask = (torch.sigmoid(logits.detach()) >= 0.5).float()
            train_dice += dice_per_image(mask, y).sum().item()

        train_loss /= len(train_set)
        train_dice /= len(train_set)
        model.eval()
        val_loss = 0
        val_dice = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = segmentation_loss(logits, y)

                val_loss += loss.item() * len(x)

                probability = torch.sigmoid(logits)
                mask = (probability >= 0.5).float()

                val_dice += dice_per_image(mask, y).sum().item()

        val_loss /= len(val_set)
        val_dice /= len(val_set)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_dice": train_dice,
            "val_dice": val_dice
        })
        if val_dice > best_dice:
            best_dice = val_dice

            torch.save({
                "model": model.state_dict(),
                "config": vars(args),
                "epoch": epoch + 1,
                "best_val_dice": best_dice
            }, save_dir / "best.pt")

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Train Dice: {train_dice:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

    torch.save({
        "model": model.state_dict(),
        "config": vars(args),
        "epoch": args.epochs,
        "best_val_dice": best_dice
    }, save_dir / "last.pt")

    write_csv(save_dir / "history.csv", history)
    plot_history(history, save_dir / "training_curves.png")

    print("Training finished!")
    print("Best Validation Dice:", best_dice)