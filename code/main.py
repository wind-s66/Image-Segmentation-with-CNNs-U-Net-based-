from arg import get_args
from argparse import Namespace
from pathlib import Path


def main():
    args = get_args()
    if args.mode == 'train':
        from trainer import train
        train(args)
        from evaluate import evaluate
        best_checkpoint = str(Path(args.save_dir) / 'best.pt')
        for mode, label in (('evaluate', 'validation evaluation'),
                            ('predict', 'test set prediction')):
            print(f'\nStarting automatic {label} with {best_checkpoint}', flush=True)
            evaluation_args = Namespace(**vars(args))
            evaluation_args.mode = mode
            evaluation_args.checkpoint = best_checkpoint
            evaluate(evaluation_args)
    else:
        from evaluate import evaluate
        evaluate(args)


if __name__ == '__main__':
    main()
