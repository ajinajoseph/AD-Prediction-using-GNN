from pathlib import Path
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
def load(p):
    try: return torch.load(p,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(p,map_location="cpu")
def main():
    total_f=total_g=0
    for cls in ["AD","MCI","CN"]:
        fs=sorted((ROOT/"features"/cls).glob("*.csv"))
        gs=sorted((ROOT/"graphs"/cls).glob("*.pt"))
        total_f+=len(fs); total_g+=len(gs)
        print(f"\\n{cls}: {len(fs)} feature CSVs, {len(gs)} graph files")
        if fs:
            df=pd.read_csv(fs[0]); print("Example CSV shape:",df.shape)
        if gs:
            g=load(gs[0]); print(f"Example graph: nodes={g.num_nodes}, features/node={g.num_node_features}, edges={g.edge_index.shape[1]}, label={g.y.tolist()}")
    print(f"\\nTotals: {total_f} feature CSVs, {total_g} graphs")
if __name__=="__main__": main()
