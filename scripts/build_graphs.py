"""Build one graph per subject. Edges are feature-similarity kNN, not anatomical connectivity."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"CN":0, "MCI":1, "AD":2}
FEATURES = ["Mean","Std","Median","Max","Min"]
K = 10

def make_edges(x):
    n = x.shape[0]
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    sd[sd < 1e-8] = 1
    z = np.nan_to_num((x-mu)/sd)
    sim = cosine_similarity(z)
    np.fill_diagonal(sim, -np.inf)
    pairs = set()
    for i in range(n):
        for j in np.argsort(sim[i])[-min(K,n-1):]:
            if i != int(j):
                pairs.add((min(i,int(j)), max(i,int(j))))
    edges, weights = [], []
    for i,j in sorted(pairs):
        w = float(np.clip((sim[i,j]+1)/2, 0, 1))
        edges.extend([[i,j],[j,i]])
        weights.extend([w,w])
    return torch.tensor(edges,dtype=torch.long).t().contiguous(), torch.tensor(weights,dtype=torch.float32).view(-1,1)

def main():
    total = 0
    for cls, label in CLASSES.items():
        source, dest = ROOT/"features"/cls, ROOT/"graphs"/cls
        dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.glob("*.csv")):
            try:
                df = pd.read_csv(path)
                x = df[FEATURES].to_numpy(dtype=np.float32)
                if x.shape != (116,5):
                    raise ValueError(f"Expected (116,5) features, got {x.shape}")
                if not np.isfinite(x).all():
                    raise ValueError("NaN or infinite features found")
                edge_index, edge_attr = make_edges(x)
                g = Data(x=torch.tensor(x), edge_index=edge_index, edge_attr=edge_attr,
                         y=torch.tensor([label],dtype=torch.long))
                g.subject_id = path.stem
                g.label_name = cls
                g.roi_names = df["ROI_Name"].astype(str).tolist() if "ROI_Name" in df else [f"ROI_{i}" for i in range(116)]
                g.roi_ids = df["ROI_ID"].astype(int).tolist() if "ROI_ID" in df else list(range(116))
                torch.save(g, dest/f"{path.stem}.pt")
                total += 1
                print(f"{cls}/{path.stem}: nodes={g.num_nodes}, features={g.num_node_features}, edges={g.edge_index.shape[1]}")
            except Exception as e:
                print(f"ERROR {cls}/{path.name}: {e}")
    print(f"Graphs saved/updated: {total}")
if __name__ == "__main__":
    main()
