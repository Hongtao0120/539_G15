# P1: data, remeshing and GNN design

Owner: Hongtao0120

Deliverables:
- Data loading scripts and the train / development / final-evaluation split (by parameter set)
- C3: remesh from predicted ϕ and transfer fields onto the new mesh, checked against the FEM solver's own remeshing
- C2 GNN architecture and graph builder
- Final integration into `final/`

Planned layout (add as the work lands):

```
P1/
├── data/      loaders, split files, small sample
├── c3/        remesh + field transfer
└── gnn/       C2 model definition and graph construction
```
