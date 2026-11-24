# P4: CNN baseline / fallback (SI-ConvNeXt)

Files: `p4.py` (training), `p4_data.py` (splits + 256x256 rasterization + Dataset),
`requirements.txt`. The split is read from `P1/data/cases.csv` (train/val/test/outer) -
do not create a separate split.

## Setup
1. `pip install -r P4/requirements.txt` (the NPS repo is cloned automatically into `P4/NPS`).
2. Data: P1's epoch cache (`<repo>/cache/<campaign>/<case>/epoch_XXX.npz`). If it does not
   exist, build it with `cd P1/data && python make_cache.py --raw-dir <raw folder> --cache-dir ../../cache --jobs 8`
   (needs h5py), where the raw folder is the ME539 folder with each dataset inside, or pass `--cache-dir` to p4.py with the filepath of the shared epoch cache.

## Run
    python P4/p4.py --prepare-only                    # once: build grid files for all splits (slow)
    python P4/p4.py --epochs 10                       # all 301 frames of every train case
    python P4/p4.py --epochs 10 --max-train-cases 8 --max-val-cases 4   # sample cases per epoch
    python P4/p4.py --epochs 2 --n-frames 21 --max-train-cases 4 --max-val-cases 2   # quick check

Training uses every consecutive frame pair (300 per case at 301 frames). To shorten an epoch,
`--max-train-cases N` draws N random cases each epoch and trains on all of their frames
(fresh draw every epoch), instead of sampling frames within cases.

Each case is rasterized once to `P4/grid_cache/<case>_g256_f0-300.npy` (about 158 MB per case,
roughly 21 GB for train+val; all four splits ~27 GB). Make sure there is sufficient disk space. Pairs are read lazily (memory-mapped), so RAM use stays small. Use `--prepare-only` on a CPU job first so the GPU job doesn't spend time rasterizing.
Outputs go to `P4/outputs/`: `best_convnext_isotropic.pt` (also stores the parameter
normalization), `history.json`, `training_plot.png`.
