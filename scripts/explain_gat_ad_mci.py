from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "gat_binary_ad_mci.pt"
SPLITS = ROOT / "results" / "binary_ad_mci" / "split_subjects.json"
OUT = ROOT / "results" / "binary_ad_mci" / "gnn_explainer"

CLASSES = ["AD", "MCI"]
DROPOUT = 0.3
EXPLAINER_EPOCHS = 100
TOP_N_NODES = 15
TOP_N_EDGES = 20


def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class GATClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = DROPOUT
        self.gat1 = GATConv(
            5, 32, heads=4, concat=True,
            dropout=DROPOUT, edge_dim=1
        )
        self.gat2 = GATConv(
            128, 32, heads=1, concat=False,
            dropout=DROPOUT, edge_dim=1
        )
        self.fc = nn.Linear(32, 2)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat2(x, edge_index, edge_attr=edge_attr))
        return self.fc(global_mean_pool(x, batch))


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not SPLITS.exists():
        raise FileNotFoundError(f"Split file not found: {SPLITS}")

    OUT.mkdir(parents=True, exist_ok=True)
    subjects_out = OUT / "subjects"
    subjects_out.mkdir(exist_ok=True)

    checkpoint = load_pt(CHECKPOINT)
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))

    model = GATClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["feature_mean"].float()
    std = checkpoint["feature_std"].float()
    std[std < 1e-6] = 1.0

    # Select only the saved test subjects, from their actual class folders.
    selected = []
    missing = []

    for subject_id in splits["test"]:
        found = []
        for cls in CLASSES:
            path = ROOT / "graphs" / cls / f"{subject_id}.pt"
            if path.exists():
                found.append((cls, path))

        if len(found) == 1:
            cls, path = found[0]
            selected.append((subject_id, cls, path))
        else:
            missing.append({
                "subject_id": subject_id,
                "status": "not found" if len(found) == 0
                          else "found in multiple class folders"
            })

    if missing:
        pd.DataFrame(missing).to_csv(
            OUT / "subjects_not_explained.csv", index=False
        )
        print(f"WARNING: {len(missing)} test subjects could not be "
              "matched uniquely. See subjects_not_explained.csv.")

    if not selected:
        raise RuntimeError(
            "No saved test subjects were matched to graphs/AD or graphs/MCI."
        )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=EXPLAINER_EPOCHS),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config={
            "mode": "multiclass_classification",
            "task_level": "graph",
            "return_type": "raw",
        },
    )

    summary_rows = []
    importance_by_actual_class = {"AD": [], "MCI": []}
    roi_labels = None

    # Try to load AAL region labels for readable output.
    # The script continues with node indices if labels cannot be loaded.
    try:
        import nibabel as nib
        import xml.etree.ElementTree as ET

        atlas_path = ROOT / "atlas" / "AAL.nii"
        xml_candidates = [
            ROOT / "atlas" / "AAL.xml",
            ROOT / "atlas" / "AAL3.xml",
        ]
        if atlas_path.exists():
            atlas_data = np.asarray(nib.load(str(atlas_path)).dataobj)
            roi_ids = sorted(int(v) for v in np.unique(atlas_data) if v > 0)
            xml_path = next((p for p in xml_candidates if p.exists()), None)
            if xml_path is not None:
                root = ET.parse(xml_path).getroot()
                names = {}
                for item in root.iter():
                    idx = item.attrib.get("index", item.attrib.get("id"))
                    name = item.attrib.get("name", item.attrib.get("label"))
                    if idx is not None and name:
                        try:
                            names[int(idx)] = name
                        except ValueError:
                            pass
                if len(roi_ids) == 116:
                    roi_labels = [
                        names.get(idx, f"ROI_{idx}") for idx in roi_ids
                    ]
    except Exception as exc:
        print(f"Atlas labels unavailable; using node indices. Detail: {exc}")

    for subject_id, actual_class, graph_path in selected:
        print(f"\nExplaining {subject_id} (actual {actual_class})")

        graph = load_pt(graph_path)
        x = graph.x.float()
        edge_index = graph.edge_index.long()
        edge_attr = getattr(graph, "edge_attr", None)

        if edge_attr is not None:
            edge_attr = edge_attr.float()
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)

        if tuple(x.shape) != (116, 5):
            print(f"Skipping unexpected feature shape: {tuple(x.shape)}")
            continue

        x = torch.nan_to_num(
            (x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0
        )
        batch = torch.zeros(x.size(0), dtype=torch.long)

        with torch.no_grad():
            logits = model(x, edge_index, edge_attr, batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_id = int(logits.argmax(dim=1).item())

        predicted_class = CLASSES[predicted_id]
        confidence = float(probabilities[predicted_id])

        print(
            f"Prediction: {predicted_class}; "
            f"P(AD)={probabilities[0]:.4f}; "
            f"P(MCI)={probabilities[1]:.4f}"
        )

        # For explanation_type="model", do not pass target to explainer().
        explanation = explainer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )

        node_mask = explanation.node_mask
        edge_mask = explanation.edge_mask

        if node_mask is None or edge_mask is None:
            print("No node or edge mask returned; skipping this subject.")
            continue

        node_scores = node_mask.detach().abs()
        if node_scores.dim() == 1:
            node_scores = node_scores.unsqueeze(-1)
        node_scores = node_scores.mean(dim=-1).cpu().numpy()
        edge_scores = edge_mask.detach().abs().cpu().numpy()

        subject_out = subjects_out / subject_id
        subject_out.mkdir(parents=True, exist_ok=True)

        node_names = (
            roi_labels if roi_labels is not None
            and len(roi_labels) == len(node_scores)
            else [f"ROI_{i}" for i in range(len(node_scores))]
        )

        node_df = pd.DataFrame({
            "node_index": np.arange(len(node_scores)),
            "region": node_names,
            "importance": node_scores,
        }).sort_values("importance", ascending=False)
        node_df.to_csv(subject_out / "node_importance.csv", index=False)

        source = edge_index[0].cpu().numpy()
        target = edge_index[1].cpu().numpy()
        edge_df = pd.DataFrame({
            "source_node": source,
            "target_node": target,
            "source_region": [node_names[i] for i in source],
            "target_region": [node_names[i] for i in target],
            "importance": edge_scores,
        }).sort_values("importance", ascending=False)
        edge_df.to_csv(subject_out / "edge_importance.csv", index=False)

        top_nodes = node_df.head(TOP_N_NODES).sort_values("importance")
        plt.figure(figsize=(10, 7))
        plt.barh(top_nodes["region"], top_nodes["importance"])
        plt.xlabel("GNNExplainer node importance")
        plt.title(f"{subject_id}: top brain-region nodes")
        plt.tight_layout()
        plt.savefig(subject_out / "top_nodes.png", dpi=180)
        plt.close()

        top_edges = edge_df.head(TOP_N_EDGES).copy()
        top_edges["edge"] = (
            top_edges["source_region"].astype(str)
            + " -> "
            + top_edges["target_region"].astype(str)
        )
        top_edges = top_edges.sort_values("importance")
        plt.figure(figsize=(11, 8))
        plt.barh(top_edges["edge"], top_edges["importance"])
        plt.xlabel("GNNExplainer edge importance")
        plt.title(f"{subject_id}: top graph edges")
        plt.tight_layout()
        plt.savefig(subject_out / "top_edges.png", dpi=180)
        plt.close()

        importance_by_actual_class[actual_class].append(node_scores)

        summary_rows.append({
            "subject_id": subject_id,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "correct": actual_class == predicted_class,
            "predicted_class_probability": confidence,
            "AD_probability": float(probabilities[0]),
            "MCI_probability": float(probabilities[1]),
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
        })

        print(f"Saved explanation to: {subject_out}")

    pd.DataFrame(summary_rows).to_csv(
        OUT / "explained_subjects_summary.csv", index=False
    )

    for cls, rows in importance_by_actual_class.items():
        if rows:
            scores = np.mean(np.stack(rows), axis=0)
            pd.DataFrame({
                "node_index": np.arange(len(scores)),
                "region": (
                    roi_labels if roi_labels is not None
                    and len(roi_labels) == len(scores)
                    else [f"ROI_{i}" for i in range(len(scores))]
                ),
                "mean_importance": scores,
                "actual_class": cls,
                "subjects_averaged": len(rows),
            }).sort_values(
                "mean_importance", ascending=False
            ).to_csv(OUT / f"mean_node_importance_{cls}.csv", index=False)

    print("\nGNNExplainer completed.")
    print(f"Subjects explained: {len(summary_rows)} / {len(splits['test'])}")
    print("Output folder:", OUT)
    print("Review explained_subjects_summary.csv first.")


if __name__ == "__main__":
    main()
