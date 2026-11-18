# P1: data and splits, remeshing and field transfer, surrogate model design, final pipeline

| File | What it is |
| --- | --- |
| `cases.csv` | All 170 cases: campaign, parameters (delta, epsilonM, D) and set |
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
cases), `me601_cache_highdelta.tar` (high-delta cases) and
`me601_cache_lhs30_fast_q1.tar` (fast-growth cases); each holds
`cache/<campaign>/<case>/epoch_XXX.npz`.

| Set | Cases |
| --- | --- |
| train | 120: 80 main LHS cases + 20 high-undercooling cases `hd_*` (delta 0.80-0.90) + 20 fast-growth cases `fg_*` |
| val | 15 (10 main + 5 fast-growth), used for every model and threshold choice (always all 15) |
| test | 15 (10 main + 5 fast-growth), final evaluation only |
| outer | 20, one parameter outside the training box; final evaluation only |

Training box: delta 0.60-0.90, epsilonM 0.0375-0.0625, D 0.5-1.5. The 30 fast-growth cases
(campaign `lhs30_fast_q1`) are a Latin hypercube in delta 0.72-0.90, D 1.10-1.50, where
growth is fastest and the model errors are largest.

## Remeshing and field transfer

Rule: refine every cell within distance 8 of the predicted phi = 0
interface, coarsen the rest by one level, then 2:1 balance
(`amr.remesh.next_mesh`). Fields move between meshes
by exact FE_Q(3) evaluation.

## Architecture

A graph autoencoder (`amr.model.GraphAutoencoder`) encodes a frame on its AMR
mesh into a 4 x 64 x 64 latent grid and decodes it on the nodes of any mesh,
so it works on the meshes built by remeshing. A residual CNN
(`amr.model.LatentStepper`), conditioned on (delta, epsilonM, D), advances
the latent one frame at a time.

```
seed phi, u on the AMR mesh (t = 0)
  |
  |   +-------------------------- encoder ---------------------------+
  v   |                                                              |
(1) GNN: message passing on the mesh --(2) mean-pool into 64x64 grid--> (3) CNN --> latent z [4 x 64 x 64]
                                                                                      |
                                        (4) CNN stepper (+ delta, epsilonM, D), one frame per call
                                                                                      v
                                                                                z(t+1), z(t+2), ...
      +-------------------------- decoder ---------------------------+                |
      |                                                              |                |
(7) GNN: message passing on the mesh <--(6) sample at the nodes-- (5) CNN --> 128x128 grid <---+
  |
  v
phi, u on the nodes of the current mesh
  |
every 10 frames: (8) new mesh from the predicted phi = 0 interface --> used for the next 10 frames

```

**Encoder (steps 1-3): AMR mesh -> latent**

1. **GNN.** Four message-passing layers on the AMR mesh. Node inputs: phi,
   scaled temperature, node type, refinement level, position features.
2. **Pooling.** Node features are averaged into a 64 x 64 grid (no trainable
   parameters). This turns the irregular mesh into an image.
3. **CNN.** One convolution compresses the image to the latent z (4 x 64 x 64).

**Time stepping (step 4): latent only**

4. **CNN stepper.** A residual CNN with the parameters (delta, epsilonM, D) as
   extra input maps. It predicts the change of z for one frame:
   z(t+1) = z(t) + dz. It never sees the mesh.

**Decoder (steps 5-7): latent -> any AMR mesh**

5. **CNN.** Upsamples z to a 128 x 128 feature grid.
6. **Sampling.** Each node of the target mesh reads features from that grid at
   its position, so any mesh can be used, including one built by remeshing.
7. **GNN.** Six message-passing layers on the target mesh sharpen the interface
   and give phi and u at every node.

**Remeshing (step 8)**

8. Every 10 frames, the next mesh is built from the predicted phi = 0
   interface. The latent is not re-encoded; the mesh only decides where the
   fields are decoded.

