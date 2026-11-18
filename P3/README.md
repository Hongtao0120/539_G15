# P3: C1 Stats Network, Ridge Baseline, and C4 Trust Check

| File | Role |
| --- | --- |
| `extract_stats.py` | Extracts ground-truth `(solid_area, tip_radius_mean, interface_length)` across all 301 frames using `P1/amr` utilities |
| `train_c1.py` | Trains the degree-3 Ridge regression baseline and C1 Stats MLP on `P1/splits.json` (`train`: 100 cases) |
| `trust_check.py` | C4 trust check comparing C1 predictions against stats extracted from C2's predicted fields (or `trust_check_inputs.json`) |
| `results/` | Committed evaluation metrics (`c1_metrics.json`) and parity plots (`c1_val_parity.png`) |

## Usage

```bash
# 1. Extract summary statistics from the unpacked cache
python P3/extract_stats.py --cache_dir /path/to/cache

# 2. Train Ridge baseline and C1 MLP
python P3/train_c1.py

# 3. Run C4 trust check on final/pipeline.ipynb output
python P3/trust_check.py --pipeline_json final/trust_check_inputs.json
```
