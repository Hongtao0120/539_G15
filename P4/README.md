# P4: CNN baseline / fallback (SI-ConvNeXt)

Files: `p4.py` (training), `p4_data.py` (splits + 256x256 rasterization + Dataset),
`requirements.txt`. The split is read from `P1/data/cases.csv` (train/val/test/outer) -
do not create a separate split.

## Setup
1. `pip install -r P4/requirements.txt` (the NPS repo is cloned automatically into `P4/NPS`).
2. Data: P1's epoch cache (`<repo>/cache/<campaign>/<case>/epoch_XXX.npz`). If it does not
   exist, build it with `cd P1/data && python make_cache.py --raw-dir <raw folder> --cache-dir ../../cache --jobs 8`
   (needs h5py), where the raw folder is the ME539 folder with each dataset inside, or pass `--cache-dir` pointing to the shared copy.

## Run
    python P4/p4.py --epochs 10                       # GPU, full train/val sets
    python P4/p4.py --epochs 3 --max-train-samples 160 --max-val-samples 40   # quick CPU check

First run rasterizes each case to `P4/grid_cache/` (slow once, then reused).
Outputs go to `P4/outputs/`: `best_convnext_isotropic.pt` (also stores the parameter
normalization), `history.json`, `training_plot.png`.
Default uses frames 0..20 (`--n-frames`); a full t=300 run needs far more memory.
