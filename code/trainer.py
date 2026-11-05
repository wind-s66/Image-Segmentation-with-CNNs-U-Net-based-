from pathlib import Path
from time import perf_counter
import torch
from torch.utils.data import DataLoader
from dataset import KneeDataset, read_rows
from model import UNet
from utils import (seed_everything, get_device, segmentation_loss, dice_per_image,
                   write_json, write_csv, plot_history)


def train(args):
    seed_everything(args.seed, args.threads)
    device = get_device(args.device)
    use_cuda = device.type == 'cuda'
    if use_cuda:
        print(f'GPU: {torch.cuda.get_device_name(device)}', flush=True)
    train_set = KneeDataset(args.train_csv, args.image_size, augment=True)
    val_set = KneeDataset(args.val_csv, args.image_size)
    sets = [{r['xrays'] for r in rows} for rows in
            (train_set.rows, val_set.rows, read_rows(args.test_csv))]
    if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Data leakage: train/validation/test image paths overlap.')
    loaders = [DataLoader(ds, batch_size=args.bs, shuffle=i == 0,
                          num_workers=args.workers, pin_memory=use_cuda)
               for i, ds in enumerate((train_set, val_set))]
    model = UNet(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4)
    output = Path(args.save_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', vars(args))
    history, best, stale, best_epoch = [], -1., 0, 0
    print(f'Device={device}; train={len(train_set)}; validation={len(val_set)}', flush=True)
    for epoch in range(1, args.epochs + 1):
        start = perf_counter()
        metrics = []
        for training, loader in zip((True, False), loaders):
            model.train(training)
            loss_total, dice_total, count = 0., 0., 0
            with torch.set_grad_enabled(training):
                for x, y in loader:
                    x = x.to(device, non_blocking=use_cuda)
                    y = y.to(device, non_blocking=use_cuda)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = segmentation_loss(logits, y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite loss')
                    if training:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0,
                                                       error_if_nonfinite=True)
                        optimizer.step()
                    loss_total += loss.item() * len(x)
                    dice_total += dice_per_image((logits.detach().sigmoid() >= args.threshold).float(), y).sum().item()
                    count += len(x)
            metrics.append((loss_total / count, dice_total / count))
        row = dict(epoch=epoch, train_loss=metrics[0][0], val_loss=metrics[1][0],
                   train_dice=metrics[0][1], val_dice=metrics[1][1],
                   lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        improved = row['val_dice'] > best
        if improved:
            best_epoch = epoch
        best, stale = (row['val_dice'], 0) if improved else (best, stale + 1)
        scheduler.step(row['val_dice'])
        checkpoint = dict(model=model.state_dict(), config=vars(args), epoch=epoch,
                          best_val_dice=best, optimizer=optimizer.state_dict())
        torch.save(checkpoint, output / 'last.pt')
        if improved:
            torch.save(checkpoint, output / 'best.pt')
        write_csv(output / 'history.csv', history)
        print(f'Epoch {epoch}/{args.epochs}: loss={row["train_loss"]:.4f} '
              f'train_Dice={row["train_dice"]:.4f} '
              f'val_loss={row["val_loss"]:.4f} val_Dice={row["val_dice"]:.4f} '
              f'best={best:.4f} patience={stale}/{args.patience} '
              f'time={perf_counter() - start:.1f}s', flush=True)
        if stale >= args.patience:
            print(f'Early stopping: no validation Dice improvement for {stale} epochs.', flush=True)
            break
    plot_history(history, output / 'training_curves.png')
    print(f'Best validation Dice (resized masks): {best:.4f} '
          f'(epoch {best_epoch}); {output / "best.pt"}')
