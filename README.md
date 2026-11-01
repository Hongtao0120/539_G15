# Solidification-Surrogate

ECE 539 (Fall 2026) team project, group 15.

A fast neural surrogate for our finite-element (FEM) phase-field simulations. Given (δ, ε_M, D) and time t, the system predicts the phase field ϕ on the adaptive mesh and three summary statistics (solid area, tip position, interface length), and flags predictions it does not trust.

## System

| Component | Type | Role |
| --- | --- | --- |
| C1 | Neural (MLP) | Predicts the three summary statistics from (δ, ε_M, D, t) |
| C2 | Neural (mesh GNN) | Predicts ϕ for 10 frames at a time on a fixed adaptive mesh |
| C3 | Classical code | Remeshes from the predicted ϕ (same refinement rule as the FEM solver) and transfers fields so C2 can continue |
| C4 | Classical code | Trust check: compares C1 with statistics computed from C2's fields |
| Fallback | Neural (CNN) | CNN on a uniform grid from [LLNL NPS](https://github.com/llnl/NPS); also the baseline for the GNN vs CNN comparison |

## Repository layout

```
.
├── P1/     data preparation, C3 remesh + field transfer, C2 GNN design
├── P2/     C2 GNN training
├── P3/     C1 stats network, ridge baseline, C4 trust check
├── P4/     CNN baseline / fallback (LLNL NPS), GNN vs CNN comparison
└── final/  integrated deliverable
```

## Team

| Workspace | GitHub | Responsibility |
| --- | --- | --- |
| P1 | [Hongtao0120](https://github.com/Hongtao0120) | Data preparation and splits; C3; C2 design; final integration |
| P2 | [achen2048](https://github.com/achen2048) | C2 GNN training and rollout fine-tuning |
| P3 | [SOUP0821](https://github.com/SOUP0821) | C1 stats network and ridge baseline; C4 trust check |
| P4 | [neilfitzsimons](https://github.com/neilfitzsimons) | CNN baseline and fallback; GNN vs CNN comparison |

The mapping of Canvas names to GitHub usernames is in the private member table of the Canvas proposal PDF.

## Working agreements

- Work in your own workspace folder. Anything that other workspaces import (data loaders, split files, shared utilities) lives in `P1/` and is changed only through a pull request.
- Data is split by whole parameter set, never by frame. The split files are produced once in `P1/` and shared by everyone.
- Large data and checkpoints are not committed. Keep only a small sample and the loading scripts in the repo.
- `final/` receives code only from merged, reviewed work.

## Milestones

| Date | Checkpoint |
| --- | --- |
| Nov 8 | Data loaders and parameter-set splits in the repo |
| Nov 15 | C1 and ridge baseline trained; CNN baseline running; C3 prototype |
| Nov 22 | Progress: first C2 result next to the CNN baseline; fallback decision |
| Nov 29 | C2 + C3 multi-block rollout; C4 threshold; evaluation protocol frozen |
| Dec 6 | Review-ready repository and slides |
| Dec 9 | Final |
