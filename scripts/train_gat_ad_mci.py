"""Separate AD-vs-MCI binary GAT experiment."""
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["AD", "MCI"]  # AD=0, MCI=1
SEED = 42
BATCH = 8
HIDDEN = 32
HEADS = 4
DROPOUT = 0.3
MAX_EPOCHS = 200
PATIENCE = 25


def load_graph(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class GATClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = DROPOUT
        self.gat1 = GATConv(
            5, HIDDEN, heads=HEADS, concat=True,
            dropout=DROPOUT, edge_dim=1
        )
        self.gat2 = GATConv(
            HIDDEN * HEADS, HIDDEN, heads=1,
            concat=False, dropout=DROPOUT, edge_dim=1
        )
        self.fc = nn.Linear(HIDDEN, 2)

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


def evaluate(model, loader, device, criterion=None):
    model.eval()
    ys, preds = [], []
    loss_sum, count = 0.0, 0

    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            logits = model(
                batch_data.x, batch_data.edge_index,
                batch_data.edge_attr, batch_data.batch
            )
            y = batch_data.y.view(-1)

            if criterion is not None:
                loss_sum += criterion(logits, y).item() * len(y)
                count += len(y)

            ys.extend(y.cpu().tolist())
            preds.extend(logits.argmax(dim=1).cpu().tolist())

    avg_loss = loss_sum / max(count, 1) if criterion is not None else None
    return np.array(ys), np.array(preds), avg_loss


def main():
    seed_all()
    graphs = []
    seen_subjects = set()

    for cls in CLASSES:
        folder = ROOT / "graphs" / cls
        if not folder.exists():
            raise FileNotFoundError(f"Missing graph folder: {folder}")

        files = sorted(folder.glob("*.pt"))
        print(f"{cls}: found {len(files)} graph files")

        for path in files:
            graph = load_graph(path)

            if tuple(graph.x.shape) != (116, 5):
                print(f"Skipping unexpected graph shape: {path}")
                continue

            subject_id = str(getattr(graph, "subject_id", path.stem))

            if subject_id in seen_subjects:
                raise ValueError(
                    f"Duplicate subject ID across AD/MCI graphs: {subject_id}"
                )
            seen_subjects.add(subject_id)

            if not torch.isfinite(graph.x).all():
                raise ValueError(f"Non-finite node features in {path.name}")

            graph.y = torch.tensor([CLASSES.index(cls)], dtype=torch.long)
            graph.subject_id = subject_id
            graphs.append(graph)

    if len(graphs) < 15:
        raise SystemExit(f"Only {len(graphs)} valid graphs loaded.")

    y = np.array([int(g.y.item()) for g in graphs])
    indices = np.arange(len(graphs))

    trainval, test = train_test_split(
        indices, test_size=0.15, random_state=SEED, stratify=y
    )
    train, val = train_test_split(
        trainval, test_size=(0.15 / 0.85),
        random_state=SEED, stratify=y[trainval]
    )

    train_x = torch.cat([graphs[int(i)].x for i in train], dim=0)
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std[std < 1e-6] = 1.0

    for graph in graphs:
        graph.x = torch.nan_to_num(
            (graph.x - mean) / std,
            nan=0.0, posinf=0.0, neginf=0.0
        )

    def make_loader(indices, shuffle=False):
        return DataLoader(
            [graphs[int(i)] for i in indices],
            batch_size=BATCH, shuffle=shuffle
        )

    train_loader = make_loader(train, shuffle=True)
    val_loader = make_loader(val)
    test_loader = make_loader(test)

    present = np.unique(y[train])
    class_weights = compute_class_weight(
        class_weight="balanced", classes=present, y=y[train]
    )
    weights = torch.ones(2)
    for cls_id, weight in zip(present, class_weights):
        weights[int(cls_id)] = float(weight)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GATClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=1e-4
    )

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    history = []

    print(
        f"\nDevice={device}; graphs={len(graphs)}; "
        f"train={len(train)}, validation={len(val)}, test={len(test)}"
    )
    print("Class mapping: AD=0, MCI=1")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss, seen = 0.0, 0

        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()

            logits = model(
                batch_data.x, batch_data.edge_index,
                batch_data.edge_attr, batch_data.batch
            )
            loss = criterion(logits, batch_data.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

            total_loss += loss.item() * batch_data.num_graphs
            seen += batch_data.num_graphs

        train_loss = total_loss / max(seen, 1)
        vy, vp, val_loss = evaluate(
            model, val_loader, device, criterion
        )
        val_f1 = f1_score(vy, vp, average="macro", zero_division=0)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1
        })

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_macro_f1={val_f1:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping. Best epoch={best_epoch}")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    model.load_state_dict(best_state)
    yt, yp, test_loss = evaluate(model, test_loader, device, criterion)

    accuracy = accuracy_score(yt, yp)
    balanced = balanced_accuracy_score(yt, yp)
    macro_f1 = f1_score(yt, yp, average="macro", zero_division=0)

    print("\nBINARY AD-vs-MCI TEST RESULTS")
    print(f"Best epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(classification_report(
        yt, yp, labels=[0, 1],
        target_names=CLASSES, zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(yt, yp, labels=[0, 1]))

    out_dir = ROOT / "results" / "binary_ad_mci"
    model_dir = ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_mean": mean,
        "feature_std": std,
        "class_names": CLASSES,
        "best_epoch": best_epoch,
        "model_config": {
            "in_channels": 5, "hidden": HIDDEN,
            "heads": HEADS, "num_classes": 2, "dropout": DROPOUT
        },
        "test_accuracy": float(accuracy),
        "test_balanced_accuracy": float(balanced),
        "test_macro_f1": float(macro_f1)
    }, model_dir / "gat_binary_ad_mci.pt")

    pd.DataFrame(history).to_csv(
        out_dir / "training_history.csv", index=False
    )
    pd.DataFrame(
        confusion_matrix(yt, yp, labels=[0, 1]),
        index=CLASSES, columns=CLASSES
    ).to_csv(out_dir / "confusion_matrix.csv")

    with open(
        out_dir / "classification_report.txt", "w", encoding="utf-8"
    ) as f:
        f.write(classification_report(
            yt, yp, labels=[0, 1],
            target_names=CLASSES, zero_division=0
        ))

    with open(
        out_dir / "split_subjects.json", "w", encoding="utf-8"
    ) as f:
        json.dump({
            "train": [str(graphs[int(i)].subject_id) for i in train],
            "validation": [str(graphs[int(i)].subject_id) for i in val],
            "test": [str(graphs[int(i)].subject_id) for i in test]
        }, f, indent=2)

    print("\nSaved separate AD-vs-MCI experiment:")
    print("models/gat_binary_ad_mci.pt")
    print("results/binary_ad_mci/")


if __name__ == "__main__":
    main()
