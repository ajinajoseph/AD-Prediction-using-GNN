from pathlib import Path
import json
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "binary_ad_mci" / "gnn_explainer"
SUMMARY = BASE / "explained_subjects_summary.csv"
OUT = BASE

def load_graph(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

if not SUMMARY.exists():
    raise FileNotFoundError(f"Summary not found: {SUMMARY}")

summary = pd.read_csv(SUMMARY)
all_rows = []

for _, row in summary.iterrows():
    sid = str(row["subject_id"])
    actual_class = str(row["actual_class"])

    graph_path = ROOT / "graphs" / actual_class / f"{sid}.pt"
    score_path = BASE / "subjects" / sid / "node_importance.csv"

    if not graph_path.exists():
        print(f"Graph not found: {graph_path}")
        continue
    if not score_path.exists():
        print(f"Node importance file not found: {score_path}")
        continue

    graph = load_graph(graph_path)
    names = list(graph.roi_names)
    roi_ids = list(graph.roi_ids)
    scores = pd.read_csv(score_path)

    if len(names) != graph.x.size(0) or len(roi_ids) != graph.x.size(0):
        print(f"ROI metadata length mismatch for {sid}; skipping.")
        continue

    scores["node_index"] = scores["node_index"].astype(int)
    if not scores["node_index"].between(0, len(names) - 1).all():
        print(f"Invalid node index for {sid}; skipping.")
        continue

    scores["ROI_Name"] = scores["node_index"].map(
        lambda i: str(names[int(i)])
    )
    scores["ROI_ID"] = scores["node_index"].map(
        lambda i: int(roi_ids[int(i)])
    )
    scores["subject_id"] = sid
    scores["actual_class"] = actual_class
    scores["predicted_class"] = str(row["predicted_class"])
    scores["correct_prediction"] = bool(row["correct"])

    columns = [
        "subject_id", "actual_class", "predicted_class",
        "correct_prediction", "node_index", "ROI_ID",
        "ROI_Name", "importance"
    ]
    scores = scores[columns]

    subject_out = BASE / "subjects" / sid / "named_node_importance.csv"
    scores.sort_values("importance", ascending=False).to_csv(
        subject_out, index=False
    )
    all_rows.append(scores)
    print(f"Saved: {subject_out}")

if not all_rows:
    raise SystemExit(
        "No named-region results were generated. "
        "Check the graph files and explanation CSVs."
    )

combined = pd.concat(all_rows, ignore_index=True)
combined.to_csv(OUT / "all_named_node_importance.csv", index=False)

# Average node importance by actual class, using matching node IDs and names.
for cls in ("AD", "MCI"):
    subset = combined[combined["actual_class"] == cls]
    if subset.empty:
        print(f"No explained subjects for {cls}")
        continue

    average = (
        subset.groupby(["node_index", "ROI_ID", "ROI_Name"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            subjects_count=("subject_id", "nunique")
        )
        .sort_values("mean_importance", ascending=False)
    )
    average.to_csv(
        OUT / f"mean_named_importance_{cls}.csv", index=False
    )
    print(f"Saved: mean_named_importance_{cls}.csv")

print("\nROI mapping completed.")
print("Output folder:", OUT)
