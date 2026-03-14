# GPThash

Long-term airborne trajectory prediction with trajectory tokenization + sequence models (Transformer / Mamba / LSTM).

This repository has been reorganized to separate source code, runnable scripts, and data files.

## Project Structure

```
GPThash/
├─ src/                            # Core modules
│  ├─ __init__.py
│  ├─ config_trAISformer.py
│  ├─ data_loader_HB_globel_v2.py
│  ├─ Focal_loss.py
│  ├─ Geohash3.py
│  ├─ metrics.py
│  ├─ models.py
│  ├─ trainers.py
│  └─ utils.py
├─ scripts/                        # Entry scripts
│  ├─ train.py
│  ├─ train_token.py
│  └─ test.py
├─ data/                           # Dataset and tokenizer files
│  ├─ quin33.sqlite
│  └─ tokenizer_3D_7+1word_blur.json
├─ results/                        # Model checkpoints / logs (generated)
├─ testimg_global/                 # Evaluation figures (generated)
└─ README.md
```

## Quick Start

Run commands from the repository root (`GPThash/`).

1. Build / refresh tokenizer

```bash
python scripts/train_token.py
```

The tokenizer will be saved to:

- `data/tokenizer_3D_7+1word_blur.json`

2. Train and evaluate

```bash
python scripts/train.py --word_size 7+1_blur --model_select model_best_better_7+1word_blur.pt --token_select data/tokenizer_3D_7+1word_blur.json --epoch 10 --n_cuda 0 --batch_size 16 --n_embd 256 --n_head 8 --retrain
```

## Notes

- Default database path is now `data/quin33.sqlite`.
- `scripts/train.py` resolves tokenizer paths relative to project root.
- Model checkpoints are saved to `results/<base_model>/`.
- Training logs are saved to `results/log/`.

## Acknowledgement

This project is inspired by trAISformer and extends trajectory tokenization to 3D airborne trajectories with velocity-aware modeling.
