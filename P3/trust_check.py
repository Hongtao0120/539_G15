"""C4 Trust Check: compares C1 stats MLP against statistics extracted from C2's predicted fields."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "P1"))
sys.path.insert(0, str(REPO_ROOT / "P3"))

from amr.pipeline import AREA, field_stats
from extract_stats import compute_interface_length
from train_c1 import StatsMLP, encode_inputs


class TrustCheckerC4:
    def __init__(self, ckpt_path, rel_thresholds=(0.10, 0.10, 0.15)):
        """
        rel_thresholds: maximum allowed relative difference (solid_area, tip_radius_mean, interface_length)
        between C1 MLP and C2 field stats before flagging a prediction as unreliable.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = StatsMLP(in_dim=4, hidden_dim=128, depth=3, out_dim=3)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.y_mean = ckpt["y_mean"]
        self.y_std = ckpt["y_std"]
        self.thresholds = np.asarray(rel_thresholds, dtype=np.float32)

    @torch.no_grad()
    def predict_c1(self, params, frame):
        """Predict [solid_area, tip_radius_mean, interface_length] from (delta, epsilonM, D, t)."""
        x_raw = np.array([[params[0], params[1], params[2], float(frame)]], dtype=np.float32)
        x_norm = torch.from_numpy(encode_inputs(x_raw))
        pred = self.model(x_norm).numpy() * self.y_std + self.y_mean
        return pred[0]

    def check_frame(self, mesh, phi_pred, params, frame):
        """Run C4 trust check directly on a predicted AMR mesh frame (mesh, phi_pred)."""
        c1_stats = self.predict_c1(params, frame)
        fstats = field_stats(mesh, phi_pred)
        c2_stats = np.array(
            [
                fstats["solid_fraction"] * AREA,
                fstats["tip_radius_mean"],
                compute_interface_length(
                    mesh["pos"].astype(np.float64),
                    mesh["cells"].astype(np.int64),
                    np.asarray(phi_pred, dtype=np.float64),
                ),
            ],
            dtype=np.float32,
        )

        rel_diff = np.abs(c2_stats - c1_stats) / np.maximum(np.abs(c1_stats), 1e-6)
        trusted = bool(np.all(rel_diff <= self.thresholds))
        return {
            "trusted": trusted,
            "c1_stats": {
                "solid_area": float(c1_stats[0]),
                "tip_radius_mean": float(c1_stats[1]),
                "interface_length": float(c1_stats[2]),
            },
            "c2_stats": {
                "solid_area": float(c2_stats[0]),
                "tip_radius_mean": float(c2_stats[1]),
                "interface_length": float(c2_stats[2]),
            },
            "rel_diff": {
                "solid_area": float(rel_diff[0]),
                "tip_radius_mean": float(rel_diff[1]),
                "interface_length": float(rel_diff[2]),
            },
        }

    def check_pipeline_json(self, json_path, cases_csv):
        """Evaluate trust_check_inputs.json emitted by final/pipeline.ipynb (uses solid_fraction & tip_radius_mean)."""
        cases_df = pd.read_csv(cases_csv).set_index("case")
        with open(json_path) as fh:
            records = json.load(fh)

        results = []
        for rec in records:
            case = rec["case"]
            frame = rec["frame"]
            params = cases_df.loc[case, ["delta", "epsilonM", "D"]].to_numpy(float)
            c1_stats = self.predict_c1(params, frame)

            c2_area = float(rec["stats_pred"]["solid_fraction"]) * AREA
            c2_tip = float(rec["stats_pred"]["tip_radius_mean"])

            diff_area = abs(c2_area - c1_stats[0]) / max(abs(c1_stats[0]), 1e-6)
            diff_tip = abs(c2_tip - c1_stats[1]) / max(abs(c1_stats[1]), 1e-6)
            trusted = (diff_area <= self.thresholds[0]) and (diff_tip <= self.thresholds[1])

            results.append(
                {
                    "case": case,
                    "frame": frame,
                    "trusted": bool(trusted),
                    "rel_diff_solid_area": float(diff_area),
                    "rel_diff_tip_radius": float(diff_tip),
                }
            )
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", default=str(REPO_ROOT / "P3" / "outputs" / "c1_mlp.pt"))
    parser.add_argument("--pipeline_json", default=None, help="Optional trust_check_inputs.json from final/")
    parser.add_argument("--cases_csv", default=str(REPO_ROOT / "P1" / "cases.csv"))
    args = parser.parse_args()

    checker = TrustCheckerC4(args.ckpt_path)
    if args.pipeline_json:
        out = checker.check_pipeline_json(args.pipeline_json, args.cases_csv)
        print(json.dumps(out[:5], indent=2))
    else:
        # Quick smoke test on sample parameters
        print("Sample C1 prediction (lhs_082, t=300):", checker.predict_c1([0.8475, 0.054375, 1.405], 300))
