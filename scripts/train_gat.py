"""Stratified subject-level GAT training/evaluation; scaler is fit on train subjects only."""
from pathlib import Path
import copy, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

ROOT=Path(__file__).resolve().parents[1]
CLASSES=["CN","MCI","AD"]
SEED=42; BATCH=8; HIDDEN=32; HEADS=4; DROPOUT=.3; MAX_EPOCHS=200; PATIENCE=25

def load(p):
    try: return torch.load(p,map_location="cpu",weights_only=False)
    except TypeError: return torch.load(p,map_location="cpu")

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

class GATClassifier(nn.Module):
    def __init__(self,in_channels=5,hidden=HIDDEN,heads=HEADS,num_classes=3,dropout=DROPOUT):
        super().__init__(); self.dropout=dropout
        self.gat1=GATConv(in_channels,hidden,heads=heads,concat=True,dropout=dropout,edge_dim=1)
        self.gat2=GATConv(hidden*heads,hidden,heads=1,concat=False,dropout=dropout,edge_dim=1)
        self.fc=nn.Linear(hidden,num_classes)
    def forward(self,x,edge_index,edge_attr=None,batch=None):
        if batch is None: batch=torch.zeros(x.size(0),dtype=torch.long,device=x.device)
        if edge_attr is not None and edge_attr.dim()==1: edge_attr=edge_attr.unsqueeze(-1)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=F.elu(self.gat1(x,edge_index,edge_attr=edge_attr))
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=F.elu(self.gat2(x,edge_index,edge_attr=edge_attr))
        return self.fc(global_mean_pool(x,batch))

def evaluate(model,loader,device,criterion=None):
    model.eval(); ys=[]; preds=[]; loss_sum=0.; n=0
    with torch.no_grad():
        for b in loader:
            b=b.to(device); logits=model(b.x,b.edge_index,b.edge_attr,b.batch); y=b.y.view(-1)
            if criterion:
                loss_sum+=criterion(logits,y).item()*len(y); n+=len(y)
            ys.extend(y.cpu().tolist()); preds.extend(logits.argmax(1).cpu().tolist())
    return np.array(ys),np.array(preds),(loss_sum/max(n,1) if criterion else None)

def main():
    seed_all()
    graphs=[]
    for cls in CLASSES:
        for p in sorted((ROOT/"graphs"/cls).glob("*.pt")):
            g=load(p)
            if tuple(g.x.shape)!=(116,5):
                print("Skipping wrong shape:",p); continue
            g.y=torch.tensor([CLASSES.index(cls)],dtype=torch.long)
            g.subject_id=getattr(g,"subject_id",p.stem); graphs.append(g)
    if len(graphs)<15: raise SystemExit(f"Only {len(graphs)} graphs loaded; build/check dataset first.")
    y=np.array([int(g.y.item()) for g in graphs]); idx=np.arange(len(graphs))
    trainval,test=train_test_split(idx,test_size=.15,random_state=SEED,stratify=y)
    train,val=train_test_split(trainval,test_size=(.15/.85),random_state=SEED,stratify=y[trainval])
    # Train-only scaler prevents test/validation leakage.
    train_x=torch.cat([graphs[i].x for i in train],dim=0)
    mean=train_x.mean(0); std=train_x.std(0,unbiased=False); std[std<1e-6]=1
    for g in graphs:
        g.x=torch.nan_to_num((g.x-mean)/std,nan=0.,posinf=0.,neginf=0.)
    def loader(ix,shuffle=False): return DataLoader([graphs[int(i)] for i in ix],batch_size=BATCH,shuffle=shuffle)
    tr,va,te=loader(train,True),loader(val),loader(test)
    cls_present=np.unique(y[train]); cw=compute_class_weight("balanced",classes=cls_present,y=y[train])
    weights=torch.ones(3)
    for c,w in zip(cls_present,cw): weights[int(c)]=float(w)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=GATClassifier().to(device); criterion=nn.CrossEntropyLoss(weight=weights.to(device))
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    best_loss=float("inf"); best_state=None; best_epoch=0; patience=0; history=[]
    print(f"Device={device}; graphs={len(graphs)}; train={len(train)}, val={len(val)}, test={len(test)}")
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); run_loss=0.; seen=0
        for b in tr:
            b=b.to(device); opt.zero_grad()
            logits=model(b.x,b.edge_index,b.edge_attr,b.batch); loss=criterion(logits,b.y.view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.); opt.step()
            run_loss+=loss.item()*b.num_graphs; seen+=b.num_graphs
        vy,vp,vl=evaluate(model,va,device,criterion)
        vf1=f1_score(vy,vp,average="macro",zero_division=0)
        history.append({"epoch":epoch,"train_loss":run_loss/max(seen,1),"val_loss":vl,"val_macro_f1":vf1})
        print(f"Epoch {epoch:03d}: train_loss={run_loss/max(seen,1):.4f}, val_loss={vl:.4f}, val_macro_f1={vf1:.4f}")
        if vl<best_loss:
            best_loss=vl; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; patience=0
        else:
            patience+=1
            if patience>=PATIENCE: print(f"Early stopping. Best epoch={best_epoch}"); break
    model.load_state_dict(best_state)
    yt,yp,tl=evaluate(model,te,device,criterion)
    acc=accuracy_score(yt,yp); bal=balanced_accuracy_score(yt,yp); mf1=f1_score(yt,yp,average="macro",zero_division=0)
    print("\\nHELD-OUT TEST (not clinical validation)")
    print(f"Best epoch={best_epoch}, loss={tl:.4f}, accuracy={acc:.4f}, balanced_accuracy={bal:.4f}, macro_F1={mf1:.4f}")
    print(classification_report(yt,yp,labels=[0,1,2],target_names=CLASSES,zero_division=0))
    (ROOT/"models").mkdir(exist_ok=True); (ROOT/"results").mkdir(exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"feature_mean":mean,"feature_std":std,
                "class_names":CLASSES,"best_epoch":best_epoch,
                "model_config":{"in_channels":5,"hidden":HIDDEN,"heads":HEADS,"num_classes":3,"dropout":DROPOUT},
                "test_accuracy":float(acc),"test_balanced_accuracy":float(bal),"test_macro_f1":float(mf1)},
               ROOT/"models"/"gat_model.pt")
    pd.DataFrame(history).to_csv(ROOT/"results"/"training_history.csv",index=False)
    pd.DataFrame(confusion_matrix(yt,yp,labels=[0,1,2]),index=CLASSES,columns=CLASSES).to_csv(ROOT/"results"/"confusion_matrix.csv")
    with open(ROOT/"results"/"split_subjects.json","w",encoding="utf-8") as f:
        json.dump({"train":[str(graphs[int(i)].subject_id) for i in train],
                   "validation":[str(graphs[int(i)].subject_id) for i in val],
                   "test":[str(graphs[int(i)].subject_id) for i in test]},f,indent=2)
    print("Saved models/gat_model.pt and results/ outputs.")
if __name__=="__main__": main()
