from pathlib import Path
import torch
import pandas as pd

ROOT = Path(".")
BASE = ROOT / "results" / "binary_ad_cn" / "gnn_explainer"
SUMMARY = BASE / "explained_subjects_summary.csv"

def load_graph(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

summary = pd.read_csv(SUMMARY)
all_rows = []

for _, row in summary.iterrows():
    sid = row["subject_id"]
    actual_class = row["actual_class"]

    graph_path = ROOT / "graphs" / actual_class / f"{sid}.pt"
    if not graph_path.exists():
        print(f"Graph not found: {graph_path}")
        continue

    graph = load_graph(graph_path)
    names = list(graph.roi_names)
    roi_ids = list(graph.roi_ids)

    score_path = BASE / "subjects" / sid / "node_importance.csv"
    scores = pd.read_csv(score_path)

    scores["ROI_Name"] = scores["node_index"].apply(
        lambda i: names[int(i)]
    )
    scores["ROI_ID"] = scores["node_index"].apply(
        lambda i: roi_ids[int(i)]
    )
    scores["actual_class"] = actual_class
    scores["predicted_class"] = row["predicted_class"]
    scores["correct_prediction"] = bool(row["correct"])

    scores = scores[
        ["subject_id", "actual_class", "predicted_class",
         "correct_prediction", "node_index", "ROI_ID",
         "ROI_Name", "importance"]
    ] if "subject_id" in scores.columns else scores.assign(
        subject_id=sid
    )[["subject_id", "actual_class", "predicted_class",
       "correct_prediction", "node_index", "ROI_ID",
       "ROI_Name", "importance"]]

    out = BASE / "subjects" / sid / "named_node_importance.csv"
    scores.sort_values("importance", ascending=False).to_csv(
        out, index=False
    )
    all_rows.append(scores)
    print(f"Saved: {out}")

if not all_rows:
    raise SystemExit("No named-region results were generated.")

combined = pd.concat(all_rows, ignore_index=True)
combined.to_csv(BASE / "all_named_node_importance.csv", index=False)

# Average scores across the explained subjects in each actual class.
for cls in ("CN", "AD"):
    subset = combined[combined["actual_class"] == cls]
    if subset.empty:
        continue

    average = (
        subset.groupby(["ROI_ID", "ROI_Name"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            subjects_count=("subject_id", "nunique")
        )
        .sort_values("mean_importance", ascending=False)
    )
    average.to_csv(BASE / f"mean_named_importance_{cls}.csv", index=False)

print("\nCompleted. Output folder:", BASE)
