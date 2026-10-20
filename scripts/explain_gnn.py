"""Explain graph predictions; creates node/edge importance CSVs using PyG GNNExplainer."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
from train_gat import GATClassifier, CLASSES

ROOT=Path(__file__).resolve().parents[1]
def load(p):
    try: return torch.load(p,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(p,map_location="cpu")

class Wrapper(torch.nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x,edge_index,edge_attr=None,batch=None):
        return self.model(x,edge_index,edge_attr,batch)

def main():
    cp=ROOT/"models"/"gat_model.pt"
    if not cp.exists(): raise SystemExit("Run scripts/train_gat.py first.")
    ck=load(cp); cfg=ck["model_config"]
    model=GATClassifier(**cfg); model.load_state_dict(ck["model_state_dict"]); model.eval()
    mean=ck["feature_mean"].float(); std=ck["feature_std"].float(); std[std<1e-6]=1
    explainer=Explainer(model=Wrapper(model),algorithm=GNNExplainer(epochs=200),
        explanation_type="model",node_mask_type="attributes",edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification",task_level="graph",return_type="raw"))
    out=ROOT/"results"/"explanations"; out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for cls in ["AD","MCI","CN"]: paths.extend(sorted((ROOT/"graphs"/cls).glob("*.pt")))
    if not paths: raise SystemExit("No graphs found; run scripts/build_graphs.py first.")
    for p in paths:
        g=load(p); x=torch.nan_to_num((g.x.float()-mean)/std,nan=0.,posinf=0.,neginf=0.)
        edge_attr=g.edge_attr.float()
        if edge_attr.dim()==1: edge_attr=edge_attr.unsqueeze(-1)
        batch=torch.zeros(x.size(0),dtype=torch.long)
        with torch.no_grad(): pred=int(model(x,g.edge_index,edge_attr,batch).argmax(1).item())
        try:
            exp=explainer(
                x=x,
                edge_index=g.edge_index,
                edge_attr=edge_attr,
                batch=batch
            )
            nm=exp.node_mask
            if nm is not None:
                nm=nm.detach().cpu().numpy()
                importance=np.mean(np.abs(nm),axis=1) if nm.ndim==2 else np.abs(nm).reshape(-1)
                names=getattr(g,"roi_names",[f"ROI_{i}" for i in range(g.num_nodes)])
                ids=getattr(g,"roi_ids",list(range(g.num_nodes)))
                pd.DataFrame({"ROI_ID":ids,"ROI_Name":names,"Importance":importance}).sort_values(
                    "Importance",ascending=False).to_csv(out/f"{p.stem}_node_importance.csv",index=False)
            em=exp.edge_mask
            if em is not None:
                em=em.detach().cpu().numpy().reshape(-1); ei=g.edge_index.cpu().numpy()
                names=getattr(g,"roi_names",[f"ROI_{i}" for i in range(g.num_nodes)])
                rows=[{"Source_ROI":names[int(ei[0,i])],"Target_ROI":names[int(ei[1,i])],
                       "Importance":float(em[i])} for i in range(min(len(em),ei.shape[1]))]
                pd.DataFrame(rows).sort_values("Importance",ascending=False).to_csv(
                    out/f"{p.stem}_edge_importance.csv",index=False)
            print(f"{p.name}: predicted {CLASSES[pred]} (explained model prediction)")
        except Exception as e:
            print(f"Explainer failed for {p.name}: {e}")
            print("Check PyTorch/PyG versions and Explainer API compatibility.")
    print("Explanation outputs:",out)
if __name__=="__main__": main()
