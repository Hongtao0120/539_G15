# P1: data and splits, remeshing and field transfer, surrogate model design, final pipeline

| File | What it is |
| --- | --- |
| `cases.csv` | All 140 cases: campaign, parameters (delta, epsilonM, D) and set |
| `splits.json` | The case split used in this folder (load it with `amr.data.load_splits`) |
| `data.ipynb` | Split check, parameter coverage, AMR graph check |
| `remesh.ipynb` | Choice of the remesh rule, mesh build and field-transfer checks |
| `amr/data.py` | Cache access, splits, FE_Q(3) mesh, graph and exact field evaluation |
| `amr/remesh.py` | Remesh rule, 2:1 balance, mesh build, field transfer |
| `amr/model.py` | Surrogate model architecture: graph autoencoder and latent stepper |
| `amr/pipeline.py` | Final model <-> remeshing loop, arm-tip metrics and trust-check statistics |
| `../final/pipeline.ipynb` | Final pipeline from the seed to t = 300 on the validation cases |

Notebooks are run as HTCondor jobs on CHTC (with `papermill`), not on the
access point.

## Data and splits

PRISMS-PF dendritic solidification on the solver's adaptive (AMR) mesh, 301
frames per case (t = 0..300) in 30 epochs of 10 frames, one mesh per epoch.
On CHTC the cache is `/staging/h/hsun382/me601_cache.tar` (main and outer
cases) and `me601_cache_highdelta.tar` (high-delta cases); each holds
`cache/<campaign>/<case>/epoch_XXX.npz`.

| Set | Cases |
| --- | --- |
| train | 100: 80 main LHS cases + 20 high-undercooling cases `hd_*` (delta 0.80-0.90) |
| val | 10, used for every model and threshold choice (always all 10) |
| test | 10, final evaluation only |
| outer | 20, one parameter outside the training box; final evaluation only |

Training box: delta 0.60-0.90, epsilonM 0.0375-0.0625, D 0.5-1.5.

## Remeshing and field transfer

Rule: refine every cell within distance 8 of the predicted phi = 0
interface, coarsen the rest by one level, then 2:1 balance
(`amr.remesh.next_mesh`). It uses only the interface location, so small bulk
errors of a predicted field do not change the mesh. Fields move between meshes
by exact FE_Q(3) evaluation.

Checks (10 validation cases, 29 transitions each): the true interface always
lies in finest cells; the new mesh has about 8% more cells than the solver's;
transfer error at most 2e-4 in phi; enthalpy change at most 4e-6 of the
latent heat; meshes rebuilt from their leaves equal the solver's exactly.

## Surrogate model (architecture)

A graph autoencoder (`amr.model.GraphAutoencoder`) encodes a frame on its AMR
mesh into a 4 x 32 x 32 latent grid and decodes it on the nodes of any mesh,
so it works on the meshes built by remeshing. A residual CNN
(`amr.model.LatentStepper`), conditioned on (delta, epsilonM, D), advances
the latent one frame at a time. Training is not part of P1.

Interface for the trained model: one folder with `autoencoder.pt`
(GraphAutoencoder state dict), `stepper.pt` (LatentStepper state dict) and
`latent_normalization.pt` (`{"mean", "std"}`, per-channel tensors of shape
[1, C, 1, 1], with z_normalized = (z - mean) / std).

## Final pipeline

`amr.pipeline.run_case` starts from the seed at t = 0 and the parameters
(delta, epsilonM, D), advances the latent frame by frame, and at the end of
every 10-frame epoch decodes on the current mesh and builds the next mesh from
the predicted phase. `final/pipeline.ipynb` runs it on the 10 validation
cases with the trained model files in `model_dir` and reports solid IoU at
t = 100, 200 and 300, phase and temperature RMSE, solid-area error, arm-tip
position error and time per case. It also writes `trust_check_inputs.json`
(solid fraction and mean tip radius of the model and solver fields) for the
trust check.
