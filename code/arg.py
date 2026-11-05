import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Femur U-Net segmentation')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'predict'], default='train')
    parser.add_argument('--checkpoint', default='checkpoints/best.pt')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--base_channels', type=int, default=16)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=12)
    parser.add_argument('--threads', type=int, default=4)

    parser.add_argument("-train_csv", "--train_csv", default="data/CSVs/train.csv")
    parser.add_argument("-val_csv", "--val_csv", default="data/CSVs/val.csv")
    parser.add_argument("-test_csv", "--test_csv", default="data/CSVs/test.csv")

    parser.add_argument("-save_dir", "--save_dir", default="checkpoints/")
    parser.add_argument("-output_dir", "--output_dir", default="results/")

    parser.add_argument("-epochs", "--epochs", type=int, default=50)
    parser.add_argument("-bs", "--bs", type=int, default=8)
    parser.add_argument("-lr", "--lr", type=float, default=1e-3)
    parser.add_argument("-wd", "--wd", type=float, default=1e-5)

    parser.add_argument("-threshold", "--threshold", type=float, default=0.5)
    parser.add_argument("-device", "--device", default="cuda",
                        help="Training/inference device (default: cuda).")

    args = parser.parse_args()
    if min(args.epochs, args.bs, args.base_channels, args.patience, args.threads) < 1 or args.image_size < 16:
        parser.error('Positive parameters required; image_size must be at least 16.')
    if args.workers < 0 or not 0 < args.threshold < 1 or args.lr <= 0 or args.wd < 0:
        parser.error('Invalid workers, threshold, lr, or wd.')
    return args
