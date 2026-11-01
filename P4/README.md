# P4: CNN baseline and fallback

Owner: neilfitzsimons

Deliverables:
- CNN on a uniform grid based on [LLNL NPS](https://github.com/llnl/NPS), trained on our data
- Controlled comparison of the adaptive-mesh GNN against the CNN: same splits, loss, optimizer, training budget and similar parameter counts; both evaluated against FEM at the same points
- Fallback path: the CNN replaces C2 + C3 if the GNN route fails

Uses the shared splits from `P1/`.
