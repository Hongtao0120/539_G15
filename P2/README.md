# P2: C2 GNN training

Owner: achen2048

Deliverables:
- Training loop for the C2 mesh GNN (loss on ϕ for the 10 frames of a block, Adam, early stopping on the development set)
- Rollout fine-tuning through C3 on short multi-block sequences
- Trained checkpoints (not committed; record where they are stored) and training logs

Uses the model and graph builder from `P1/` and the shared splits.
