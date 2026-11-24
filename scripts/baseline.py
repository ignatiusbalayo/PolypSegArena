"""Baseline for PolypSegArena: a plain 3-layer GCN on the superpixel graphs (pure PyTorch, CPU).

    python scripts/baseline.py
    python scripts/baseline.py --epochs 20 --limit 40      # quick smoke test

Every training image is a graph; all graphs are stacked into one big disconnected graph and
trained full-batch. Node features are standardised; the target is label_frac (soft label).
Writes predictions.csv (id,y_pred). Validate before encrypting:
    python competition/validate_submission.py predictions.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "public"
META = {"image_id", "node_id", "id", "label_frac", "label"}


def load_graph(nodes_csv: str, edges_csv: str, limit: int = 0):
    nodes = pd.read_csv(PUB / nodes_csv)
    edges = pd.read_csv(PUB / edges_csv)
    if limit:
        keep = sorted(nodes["image_id"].unique())[:limit]
        nodes, edges = nodes[nodes["image_id"].isin(keep)], edges[edges["image_id"].isin(keep)]
    nodes = nodes.reset_index(drop=True)
    feat_cols = [c for c in nodes.columns if c not in META]
    index = {(i, n): k for k, (i, n) in enumerate(zip(nodes["image_id"], nodes["node_id"]))}
    src = [index[(i, s)] for i, s in zip(edges["image_id"], edges["source"])]
    dst = [index[(i, t)] for i, t in zip(edges["image_id"], edges["target"])]
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)  # undirected
    return nodes, feat_cols, ei


def norm_adj(ei: torch.Tensor, n: int) -> torch.Tensor:
    loops = torch.arange(n).repeat(2, 1)
    ei = torch.cat([ei, loops], 1)
    deg = torch.zeros(n).index_add_(0, ei[0], torch.ones(ei.shape[1]))
    w = deg[ei[0]].rsqrt() * deg[ei[1]].rsqrt()
    return torch.sparse_coo_tensor(ei, w, (n, n)).coalesce()


class GCN(nn.Module):
    def __init__(self, d_in, hidden=64, layers=3, dropout=0.3):
        super().__init__()
        dims = [d_in] + [hidden] * (layers - 1) + [1]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.dropout = dropout

    def forward(self, x, a):
        for i, lin in enumerate(self.lins):
            x = torch.sparse.mm(a, lin(F.dropout(x, self.dropout, self.training)))
            if i < len(self.lins) - 1:
                x = F.relu(x)
        return x.squeeze(-1)


def dice_of(logits, nodes, offset=0.0):
    """Mean per-image Dice of painted node predictions (uses area and label_frac, as the scorer does)."""
    on = (logits.detach().numpy() + offset >= 0).astype(float)
    d = pd.DataFrame({"i": nodes["image_id"].to_numpy(), "a": nodes["area"].to_numpy(), "f": nodes["label_frac"].to_numpy(), "on": on})
    d["gt"], d["pr"] = d["a"] * d["f"], d["a"] * d["on"]
    d["tp"] = d["gt"] * d["on"]
    g = d.groupby("i")[["tp", "pr", "gt"]].sum()
    return float((2 * g["tp"] / (g["pr"] + g["gt"]).clip(lower=1e-9)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N train graphs")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    tr, cols, tr_ei = load_graph("train_nodes.csv", "train_edges.csv", args.limit)
    te, _, te_ei = load_graph("test_nodes.csv", "test_edges.csv")
    X = tr[cols].to_numpy(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xtr = torch.from_numpy((X - mu) / sd)
    Xte = torch.from_numpy((te[cols].to_numpy(np.float32) - mu) / sd)
    ytr = torch.from_numpy(tr["label_frac"].to_numpy(np.float32))
    A_tr, A_te = norm_adj(tr_ei, len(tr)), norm_adj(te_ei, len(te))

    # hold out 15% of train graphs to tune the decision offset
    ids = sorted(tr["image_id"].unique())
    rng = np.random.default_rng(0)
    val_ids = set(rng.choice(ids, max(1, int(0.15 * len(ids))), replace=False))
    is_val = torch.from_numpy(tr["image_id"].isin(val_ids).to_numpy())
    pos_w = torch.tensor(float((1 - ytr[~is_val].mean()) / ytr[~is_val].mean()))
    print(f"positive-class weight {pos_w.item():.2f}")

    model = GCN(len(cols), args.hidden, args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        out = model(Xtr, A_tr)
        loss = F.binary_cross_entropy_with_logits(out[~is_val], ytr[~is_val], pos_weight=pos_w)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (epoch + 1) % 25 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vd = dice_of(model(Xtr, A_tr)[is_val], tr[is_val.numpy()])
            print(f"epoch {epoch + 1}/{args.epochs}  loss {loss.item():.4f}  val_dice {vd:.4f}  elapsed {(time.time() - t0) / 60:.2f} min")

    model.eval()
    with torch.no_grad():
        val_logits = model(Xtr, A_tr)[is_val]
        offsets = np.linspace(-3, 3, 25)
        scores = [dice_of(val_logits, tr[is_val.numpy()], o) for o in offsets]
        best = float(offsets[int(np.argmax(scores))])
        print(f"validation Dice {max(scores):.4f} at logit offset {best:+.2f} (offset 0: {dice_of(val_logits, tr[is_val.numpy()]):.4f})")
        prob = torch.sigmoid(model(Xte, A_te) + best).numpy()
    pd.DataFrame({"id": te["id"], "y_pred": np.round(prob, 5)}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}. Total time {(time.time() - t0) / 60:.2f} min")


if __name__ == "__main__":
    main()
