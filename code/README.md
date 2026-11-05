# Femur Segmentation with U-Net

Binary femur segmentation from knee X-ray images using a four-level U-Net.

## Project structure

```text
femur_project/
  arg.py            # Command-line arguments
  dataset.py        # CSV loading, preprocessing and augmentation
  model.py          # U-Net architecture
  trainer.py        # Training, validation and checkpoint saving
  evaluate.py       # Evaluation, test predictions and visualization
  utils.py          # Dice, loss, reproducibility and output utilities
  main.py           # Entry point and automatic train/evaluate/predict pipeline
  requirements.txt
  data/
    CSVs/           # train.csv, val.csv, test.csv, dataset.csv
    train/xrays/    # 100 annotated X-ray images
    train/masks/    # Corresponding binary masks
    test/xrays/     # 50 test images
```

## Run

Use your existing Python environment with CUDA-enabled PyTorch. Python 3.10 or
later is required. Select that interpreter in your editor and terminal.
The dependencies are listed in `requirements.txt`; install missing dependencies
with `python -m pip install -r requirements.txt` if needed.

```bash
python main.py --mode train
```

Training uses the NVIDIA GPU by default. After training finishes, including early
stopping, the program automatically loads the best model, evaluates the validation
set and processes all 50 test images. No separate test command is required.

Run again without retraining:

```bash
python main.py --mode evaluate
python main.py --mode predict
```

Useful options:

```bash
python main.py --epochs 50 --bs 8 --lr 0.001
python main.py --mode predict --checkpoint checkpoints/best.pt
python main.py --mode predict --device cpu
```

## Data and method

- Split: 80 training images, 20 validation images, 50 test images.
- CSV columns: `xrays,masks`; paths are relative to the project root.
- Unlabeled test images have `None` in the `masks` column.
- Masks: white = femur, black = background; image/mask dimensions must match.
- Input: grayscale, normalized to [0, 1], resized to 256 x 256.
- Training augmentation: synchronized horizontal flips and rotations up to 10 degrees.
- Loss: binary cross-entropy with logits + soft Dice loss.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.00001, batch size 8.
- Gradient clipping: maximum norm 1.0, to limit unusually large parameter updates.
- Maximum 50 epochs; early stopping after 12 epochs without validation Dice improvement.
- Best checkpoint selected using validation hard Dice, with threshold 0.5.

Independent evaluation restores predictions to the original image dimensions.
Dice is calculated per image and averaged. Both-empty masks receive Dice 1.
The current test set has no ground-truth masks, so it produces segmentation
outputs without a test Dice score. If test masks are supplied in `test.csv`,
their Dice scores are calculated automatically.

## Outputs

| Location | Contents |
| --- | --- |
| `checkpoints/best.pt` | Best validation checkpoint |
| `checkpoints/last.pt` | Last training checkpoint |
| `checkpoints/config.json` | Run parameters |
| `checkpoints/history.csv` | Training history |
| `checkpoints/training_curves.png` | Loss and validation Dice curves |
| `results/evaluate/` | Validation predictions and Dice results |
| `results/predict/` | Predictions for the 50 test images |

Each result directory contains `masks/`, `overlays/`, `comparisons/`,
`per_image_dice.csv` and `metrics.json`. Comparison panels show the X-ray,
then ground truth (when available), prediction and red overlay.
Each panel has a title. The terminal shows test progress every 10 images and
clearly distinguishes validation Dice from test output. Training logs include
training/validation Dice, the best score, early-stopping patience and epoch time.

## GitHub submission

Upload the seven Python modules, `README.md`, `requirements.txt`, `.gitignore`
and the CSV split files. Raw images, model weights, generated results and editor
settings are excluded by `.gitignore` and remain on your computer.
Place the course dataset in the structure above when cloning the repository.

Submit your repository link and a full-screen video of at most five minutes,
showing the code, evaluation, example segmentations and a brief explanation.
