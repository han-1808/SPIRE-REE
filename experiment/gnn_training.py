import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "preprocessing"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import Identity, Linear
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
            self.bns.append(torch.nn.BatchNorm1d(dims[i + 1]))
            self.skips.append(
                Linear(dims[i], dims[i + 1], bias=False) if dims[i] != dims[i + 1] else Identity()
            )
        self.convs.append(SAGEConv(dims[-2], dims[-1]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.bns[i](h)
            x = F.relu(h + self.skips[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(dims[i], dims[i + 1]))
            self.bns.append(torch.nn.BatchNorm1d(dims[i + 1]))
            self.skips.append(
                Linear(dims[i], dims[i + 1], bias=False) if dims[i] != dims[i + 1] else Identity()
            )
        self.convs.append(GCNConv(dims[-2], dims[-1]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index, edge_weight)
            h = self.bns[i](h)
            x = F.relu(h + self.skips[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index, edge_weight)


class GAT(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.skips = torch.nn.ModuleList()

        head_dim = max(1, hidden_dim // heads)
        actual_hidden = head_dim * heads

        dims = [in_dim] + [actual_hidden] * (num_layers - 1) + [out_dim]
        for i in range(num_layers - 1):
            self.convs.append(GATConv(dims[i], head_dim, heads=heads, dropout=dropout, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(dims[i + 1]))
            self.skips.append(
                Linear(dims[i], dims[i + 1], bias=False) if dims[i] != dims[i + 1] else Identity()
            )
        self.convs.append(GATConv(dims[-2], out_dim, heads=1, dropout=dropout, concat=False))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            h = conv(x, edge_index)
            h = self.bns[i](h)
            x = F.relu(h + self.skips[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def augment_with_edge_stats(
    X_node: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Append per-node mean and max of incoming edge weights as 2 extra features.

    Returns a new tensor of shape (N, F+2).  Nodes with no incoming edges get
    0.0 for both statistics.  Uses numpy ufunc.at so it works on any PyTorch
    version without scatter_reduce_.
    """
    N      = X_node.shape[0]
    ew_np  = edge_weight.detach().cpu().numpy().astype(np.float32)
    dst_np = edge_index[1].detach().cpu().numpy()

    ew_sum = np.zeros(N, dtype=np.float32)
    count  = np.zeros(N, dtype=np.float32)
    np.add.at(ew_sum, dst_np, ew_np)
    np.add.at(count,  dst_np, 1.0)
    mean_ew = ew_sum / np.maximum(count, 1e-8)

    max_ew = np.zeros(N, dtype=np.float32)
    np.maximum.at(max_ew, dst_np, ew_np)

    dev    = X_node.device
    mean_t = torch.from_numpy(mean_ew).unsqueeze(1).to(dev)
    max_t  = torch.from_numpy(max_ew).unsqueeze(1).to(dev)
    return torch.cat([X_node.to(dtype=torch.float32), mean_t, max_t], dim=1)


def build_pyg_data(X_node, edge_index, edge_weight, y) -> Data:
    return Data(
        x=X_node,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor(y.values, dtype=torch.long),
    )


def induce_subgraph(data, node_idx):
    mapping = torch.full((data.num_nodes,), -1, dtype=torch.long, device=data.edge_index.device)
    mapping[node_idx] = torch.arange(node_idx.numel(), dtype=torch.long, device=data.edge_index.device)

    src = data.edge_index[0]
    dst = data.edge_index[1]
    edge_mask = (mapping[src] >= 0) & (mapping[dst] >= 0)

    sub_edge_index = mapping[data.edge_index[:, edge_mask]]
    sub_edge_weight = data.edge_weight[edge_mask]
    sub_x = data.x[node_idx]
    sub_y = data.y[node_idx]

    return Data(
        x=sub_x,
        edge_index=sub_edge_index,
        edge_weight=sub_edge_weight,
        y=sub_y,
    )


def build_train_val_subgraph(data, train_idx, val_idx):
    train_val_idx = torch.cat([train_idx, val_idx], dim=0)
    train_val_data = induce_subgraph(data, train_val_idx)

    sub_train_idx = torch.arange(train_idx.numel(), dtype=torch.long, device=data.x.device)
    sub_val_idx = torch.arange(
        train_idx.numel(),
        train_idx.numel() + val_idx.numel(),
        dtype=torch.long,
        device=data.x.device,
    )
    return train_val_data, sub_train_idx, sub_val_idx


def train_epoch(model, data, optimizer, criterion, train_idx):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[train_idx], data.y[train_idx])

    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_metrics(model, data, idx, threshold=0.5):
    model.eval()

    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=1)[:, 1]

    preds = (probs >= threshold).long()

    y_true = data.y[idx].cpu().numpy()
    y_pred = preds[idx].cpu().numpy()
    y_prob = probs[idx].cpu().numpy()

    has_both_classes = len(np.unique(y_true)) > 1
    single_score = 1.0 if np.all(y_pred == y_true) else 0.0
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob) if has_both_classes else single_score,
        "pr_auc":    average_precision_score(y_true, y_prob) if has_both_classes else single_score,
    }

    return metrics


def tune_gnn_model(
    model_class,
    data,
    train_idx,
    val_idx,
    device,
    hidden_dims=None,
    num_layers_list=None,
    dropouts=None,
    lrs=None,
    return_records=False,
):
    search_space = {
        "hidden_dim": hidden_dims if hidden_dims is not None else [32, 64, 128],
        "num_layers": num_layers_list if num_layers_list is not None else [1, 2],
        "dropout": dropouts if dropouts is not None else [0.3, 0.5],
        "lr": lrs if lrs is not None else [0.01, 0.005, 0.001],
    }
    best_f1 = 0
    best_config = None
    records = []

    for hidden_dim in search_space["hidden_dim"]:
        for num_layers in search_space["num_layers"]:
            for dropout in search_space["dropout"]:
                for lr in search_space["lr"]:
                    model = model_class(
                        in_dim=data.x.shape[1],
                        hidden_dim=hidden_dim,
                        out_dim=2,
                        num_layers=num_layers,
                        dropout=dropout,
                    ).to(device)

                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=lr,
                        weight_decay=5e-4,
                    )

                    criterion = torch.nn.CrossEntropyLoss()

                    best_state = None
                    best_f1_inner = 0.0
                    for _ in range(100):
                        train_epoch(model, data, optimizer, criterion, train_idx)
                        f1_inner = evaluate_metrics(model, data, val_idx)["f1"]
                        if f1_inner > best_f1_inner:
                            best_f1_inner = f1_inner
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    if best_state is not None:
                        model.load_state_dict(best_state)

                    val_metrics = evaluate_metrics(model, data, val_idx)
                    val_f1 = val_metrics["f1"]

                    print(
                        f"hidden={hidden_dim}, layers={num_layers}, "
                        f"dropout={dropout}, lr={lr} -> Val F1={val_f1:.3f}"
                    )

                    records.append({
                        "hidden_dim": hidden_dim,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "lr": lr,
                        "val_f1": float(val_f1),
                        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
                    })

                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_config = (hidden_dim, num_layers, dropout, lr)

    print("Best config:", best_config)
    print("Best Val F1:", best_f1)
    if return_records:
        return best_config, best_f1, records
    return best_config, best_f1


def train_gnn_model(
    model_class,
    data,
    train_idx,
    val_idx,
    test_idx,
    hidden_channels,
    num_layers,
    dropout,
    lr,
    epochs,
    patience,
    device,
    criterion,
):
    model = model_class(
        in_dim=data.x.shape[1],
        hidden_dim=hidden_channels,
        out_dim=2,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
    )

    history = {"train_loss": [], "val_f1": [], "test_f1": []}
    best_val_f1 = 0
    best_model_state = None
    epochs_no_improve = 0

    print(
        "Training GNN model with "
        f"hidden_channels={hidden_channels}, num_layers={num_layers}, "
        f"dropout={dropout}, lr={lr}"
    )

    for epoch in range(epochs):
        train_loss = train_epoch(model, data, optimizer, criterion, train_idx)
        val_metrics = evaluate_metrics(model, data, val_idx)
        test_metrics = evaluate_metrics(model, data, test_idx) if test_idx is not None else None

        history["train_loss"].append(train_loss)
        history["val_f1"].append(val_metrics["f1"])
        if test_metrics is not None:
            history["test_f1"].append(test_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            if test_metrics is not None:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val F1: {val_metrics['f1']:.3f}, Test F1: {test_metrics['f1']:.3f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val F1: {val_metrics['f1']:.3f}"
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_test_metrics = evaluate_metrics(model, data, test_idx) if test_idx is not None else None
    if final_test_metrics is not None:
        print(f"Final Test Metrics: {final_test_metrics}")

    return model, history, final_test_metrics
