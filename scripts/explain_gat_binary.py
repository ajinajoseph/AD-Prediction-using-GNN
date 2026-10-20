from pathlib import Path
import json
import csv

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
CHECKPOINT = ROOT / "models" / "gat_binary_ad_cn.pt"
SPLITS = ROOT / "results" / "binary_ad_cn" / "split_subjects.json"
OUT = ROOT / "results" / "binary_ad_cn" / "gnn_explainer"

CLASSES = ["CN", "AD"]
DROPOUT = 0.3
MAX_SUBJECTS_PER_CLASS = 100
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "subjects").mkdir(exist_ok=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")
    if not SPLITS.exists():
        raise FileNotFoundError(f"Split file not found: {SPLITS}")

    checkpoint = load_pt(CHECKPOINT)
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))

    model = GATClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["feature_mean"].float()
    std = checkpoint["feature_std"].float()
    std[std < 1e-6] = 1.0

    # Find test subjects by their saved IDs and actual class folders.
    selected = []
    for actual_class in CLASSES:
        folder = ROOT / "graphs" / actual_class
        available = {p.stem: p for p in folder.glob("*.pt")}
        count = 0

        for subject_id in splits["test"]:
            if subject_id in available:
                selected.append(
                    (subject_id, actual_class, available[subject_id])
                )
                count += 1
                if count >= MAX_SUBJECTS_PER_CLASS:
                    break

    if not selected:
        raise RuntimeError(
            "No test subjects from split_subjects.json were found "
            "in graphs/AD or graphs/CN."
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
    node_importance_by_class = {"CN": [], "AD": []}

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
            print(f"Skipping unexpected shape for {subject_id}: {x.shape}")
            continue

        # Apply exactly the feature normalization saved during training.
        x = torch.nan_to_num(
            (x - mean) / std,
            nan=0.0, posinf=0.0, neginf=0.0
        )
        batch = torch.zeros(x.size(0), dtype=torch.long)

        with torch.no_grad():
            logits = model(x, edge_index, edge_attr, batch)
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_id = int(logits.argmax(dim=1).item())

        predicted_class = CLASSES[predicted_id]
        confidence = float(probabilities[predicted_id])

        print(f"Prediction: {predicted_class}")
        print(f"Probabilities: CN={probabilities[0]:.4f}, "
              f"AD={probabilities[1]:.4f}")

        explanation = explainer(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
        )

        node_mask = explanation.node_mask
        edge_mask = explanation.edge_mask

        if node_mask is None or edge_mask is None:
            print("No node or edge masks returned; skipping.")
            continue

        # One score per node: aggregate feature attribution magnitudes.
        node_scores = node_mask.detach().abs()
        if node_scores.dim() == 1:
            node_scores = node_scores.unsqueeze(-1)
        node_scores = node_scores.mean(dim=-1).cpu().numpy()

        edge_scores = edge_mask.detach().abs().cpu().numpy()

        subject_out = OUT / "subjects" / subject_id
        subject_out.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "node_index": np.arange(len(node_scores)),
            "importance": node_scores,
        }).sort_values(
            "importance", ascending=False
        ).to_csv(subject_out / "node_importance.csv", index=False)

        source = edge_index[0].cpu().numpy()
        target = edge_index[1].cpu().numpy()

        edge_df = pd.DataFrame({
            "source_node": source,
            "target_node": target,
            "importance": edge_scores,
        }).sort_values("importance", ascending=False)

        edge_df.to_csv(subject_out / "edge_importance.csv", index=False)

        # Plot top nodes; node indices can later be mapped to AAL labels.
        top_nodes = pd.DataFrame({
            "node": np.arange(len(node_scores)),
            "importance": node_scores,
        }).nlargest(TOP_N_NODES, "importance").sort_values("importance")

        plt.figure(figsize=(9, 6))
        plt.barh(top_nodes["node"].astype(str), top_nodes["importance"])
        plt.xlabel("GNNExplainer node importance")
        plt.ylabel("AAL ROI index (0-based)")
        plt.title(f"{subject_id}: top brain-region nodes")
        plt.tight_layout()
        plt.savefig(subject_out / "top_nodes.png", dpi=180)
        plt.close()

        # Plot top directed edges as stored in the graph.
        top_edges = edge_df.head(TOP_N_EDGES).copy()
        top_edges["edge"] = (
            top_edges["source_node"].astype(str)
            + " -> "
            + top_edges["target_node"].astype(str)
        )
        top_edges = top_edges.sort_values("importance")

        plt.figure(figsize=(10, 7))
        plt.barh(top_edges["edge"], top_edges["importance"])
        plt.xlabel("GNNExplainer edge importance")
        plt.ylabel("Graph edge (source -> target)")
        plt.title(f"{subject_id}: top graph edges")
        plt.tight_layout()
        plt.savefig(subject_out / "top_edges.png", dpi=180)
        plt.close()

        node_importance_by_class[actual_class].append(node_scores)

        summary_rows.append({
            "subject_id": subject_id,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
            "correct": actual_class == predicted_class,
            "predicted_class_probability": confidence,
            "CN_probability": float(probabilities[0]),
            "AD_probability": float(probabilities[1]),
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
        })

        print(f"Saved explanation: {subject_out}")

    pd.DataFrame(summary_rows).to_csv(
        OUT / "explained_subjects_summary.csv", index=False
    )

    # Exploratory class-wise averages across the explained subjects.
    for cls, rows in node_importance_by_class.items():
        if rows:
            scores = np.mean(np.stack(rows), axis=0)
            pd.DataFrame({
                "node_index": np.arange(len(scores)),
                "mean_importance": scores,
                "actual_class": cls,
                "subjects_averaged": len(rows),
            }).sort_values(
                "mean_importance", ascending=False
            ).to_csv(OUT / f"mean_node_importance_{cls}.csv", index=False)

    print("\nGNNExplainer completed.")
    print("Output folder:", OUT)
    print("Review explained_subjects_summary.csv first.")


if __name__ == "__main__":
    main()
