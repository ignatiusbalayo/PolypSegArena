# PolypSegArena - Segmentation Challenge

## 🎯 Challenge Overview
**Can your graph neural network segment polyps in colonoscopy images, under noise and a shifted polyp-size distribution, using only a CPU?**

PolypSegArena is a **graph-learning** competition built on a modified version of
[Kvasir-SEG](https://datasets.simula.no/kvasir-seg/). Each colonoscopy image is turned into a graph:
the nodes are SLIC superpixels, edges connect superpixels that touch, and every node carries 17
hand-crafted features. **No raw pixels are published.** You predict, for every node of every test
graph, the probability that it belongs to a polyp. Painting your node predictions back onto the image
gives a mask, and the score is the **mean per-image Dice coefficient** (mean IoU is secondary).

The competition is part of a GNN benchmark that compares LLM coding agents with human participants.
Every submission is tagged `human`, `llm` or `hybrid`. **Your model must be a graph neural network
that uses the edges** (see Rules).

---

## 📊 Dataset Information

Derived from Kvasir-SEG (1,000 polyp images with masks) and **modified so it differs from the original**. Each image became one graph with the following properties:

* **Undirected:** True
* **Self-loops:** False
* **Total Graphs:** 922 (522 train + 400 test)
* **Total Nodes:** 82,126 (46,420 train + 35,706 test), about 89 per graph
* **Total Node Features:** 17
* **Total Edges:** 211,912 (119,821 train + 92,091 test)

* **Task:** node classification (polyp vs. background superpixel), scored as image segmentation
* **Graphs:** one per image; about 100 nodes each (see `data/public/dataset_stats.json`)
* **Node features (17):** superpixel area, mean and std of R/G/B, hue (cos, sin), mean saturation and value, mean edge strength, centroid (cx, cy), bounding-box width and height, fill ratio
* **Edges:** undirected region adjacency, listed once per pair
* **Graph ids:** shuffled and renamed `img_0001...` (original file names are hidden)
* **Train / test:** 400 test graphs with the natural distribution, the rest forms the train pool

### Modifications (applied to the images before the graphs are built)
| Modification | Where | Effect |
|:--|:--|:--|
| Size imbalance | train only | Polyps in the largest area quartile are kept with probability 0.5, so small polyps dominate training. The test set keeps the natural size distribution. |
| Geometric changes | all images, stronger on test | Random crop-and-resize, horizontal / vertical flip and 90-degree rotation. Masks are transformed with the image, and a crop must keep at least 90% of the polyp. |
| Colour jitter | all images, stronger on test | Random brightness, contrast, saturation and hue shifts. |
| Image noise | 50% of test, 20% of train (milder) | Gaussian noise, Gaussian blur or heavy JPEG compression. |
| Label noise | 10% of train masks | Masks randomly dilated or eroded by 2-5 px before node labels are computed. Test labels are never perturbed. |

### Protection against matching the original source
Kvasir-SEG is public, so a model or agent could try to recognise a test image and look up its
original mask. To make that harder, the geometric changes, colour jitter and noise above were
applied to the images, every graph was renamed in shuffled order, the train/test split is new, and
only superpixel graphs with summary features are released (no pixels). These steps **reduce** the
chance that a graph can be matched to its public source. They do **not** make it impossible, and a
model that saw Kvasir-SEG during pretraining may still have an advantage. Participants must not look
up the original dataset (see Rules).

---

## 🔥 What Makes This Hard?

- ✅ **Class and size imbalance** - small polyps dominate training, while the test set has more large polyps
- ✅ **Noisy test images** - noise, blur and compression change node colour and texture in ways rarely seen in training
- ✅ **Imperfect training labels** - 10% of training masks have shifted boundaries
- ✅ **Graph structure matters** - a node's colour alone is ambiguous; neighbouring superpixels tell you whether it is part of a polyp
- ✅ **Superpixel ceiling** - superpixel edges do not follow the polyp edge exactly, so even perfect node labels score 0.8845

**Expected Performance (mean Dice):**
- All-background submission: 0.00
- Predict every node as polyp: ~0.30
- Plain GCN baseline: ~0.38
- Good GNN: >0.5 🎯
- Ceiling (perfect node labels): 0.8845

---

## 📈 Dataset Distribution

| Split | Graphs | Nodes | Edges | Polyp-node rate |
|:------|:-------|:------|:------|:----------------|
| **Training** | 522 | 46,420 | 119,821 | 14.2% |
| **Testing** | 400 | 35,706 | 92,091 | 19.1% (labels hidden) |

Numbers come from `data/public/dataset_stats.json`.

------|:------|:------|
| Train | see stats | `train_nodes.csv` (features + labels), `train_edges.csv` |
| Test  | 400 | `test_nodes.csv` (features only), `test_edges.csv`; labels hidden (`private/test_labels.csv`, encrypted copy in `encryption/`) |

---

## 📋 Quick Links

- **Submit:** open a Pull Request (see section 4)
- **Leaderboard:** https://ignatiusbalayo.github.io/PolypSegArena/leaderboard.html
- **Download dataset:** https://github.com/ignatiusbalayo/PolypSegArena/releases/download/v1.0/PolypSegArena_Dataset.zip
- **Submissions guide:** [submissions/README.md](submissions/README.md)

---

## 1. Repository Structure
```
.
├── data/
│   ├── public/
│   │   ├── train_nodes.csv             # image_id, node_id, 17 features, label_frac, label
│   │   ├── train_edges.csv             # image_id, source, target
│   │   ├── test_nodes.csv              # id, image_id, node_id, 17 features (no labels)
│   │   ├── test_edges.csv              # image_id, source, target
│   │   ├── dataset_stats.json
│   │   └── sample_submission.csv       # id, y_pred
│   ├── raw/                            # original Kvasir-SEG (git-ignored)
│   └── private/                        # staging images + test_labels.csv (git-ignored)
├── encryption/
│   ├── public_key.pem                  # participants encrypt with this
│   └── test_labels.enc                 # encrypted private labels
├── competition/
│   ├── config.yaml
│   ├── validate_submission.py
│   ├── evaluate.py
│   ├── metrics.py                      # node-to-pixel Dice / IoU
│   ├── crypto_utils.py
│   ├── score_pr.py                     # CI orchestrator
│   └── render_leaderboard.py
├── leaderboard/                        # leaderboard.csv / leaderboard.md
├── docs/                               # GitHub Pages leaderboard
├── scripts/
│   ├── baseline.py                     # 3-layer GCN baseline (pure PyTorch)
│   ├── encrypt_submission.py           # participants
│   ├── prepare_data.py                 # organizers (modify images)
│   ├── build_graphs.py                 # organizers (images -> graphs)
│   ├── generate_keys.py                # organizers
│   ├── encrypt_labels.py               # organizers
│   └── make_release.py                 # organizers
├── submissions/inbox/                  # one folder per team (via Pull Request)
├── ORGANIZERS.md                       # setup checklist for the organizers
└── .github/workflows/score_submission.yml
```

---

## 2. Dataset Files

### train_nodes.csv
```csv
image_id,node_id,area,r_mean,...,fill_ratio,label_frac,label
img_0001,0,0.0102,0.61,...,0.55,0.0,0
```
`label_frac` is the share of the node's pixels inside the polyp; `label` = 1 when `label_frac > 0.5`.
Use either as the training target.

### train_edges.csv / test_edges.csv
```csv
image_id,source,target
img_0001,0,1
```
`source` and `target` are `node_id`s inside the same graph. Each undirected pair appears once.

### test_nodes.csv
Same columns without `label_frac` / `label`, plus a global `id` that you must use in your submission.

**See `data/public/sample_submission.csv` for the required output format.**

---

## 3. 📥 Download & Setup
```bash
git clone https://github.com/ignatiusbalayo/PolypSegArena.git
cd PolypSegArena

# Download dataset (GitHub Release)
wget https://github.com/ignatiusbalayo/PolypSegArena/releases/download/v1.0/PolypSegArena_Dataset.zip
unzip PolypSegArena_Dataset.zip

pip install -r requirements.txt

# Run the baseline (CPU)
python scripts/baseline.py
```

---

## 4. 📤 How to Submit

Submissions are **encrypted** so nobody can copy another team's predictions.

1. Create `predictions.csv` (format below) and validate it:
   ```bash
   python competition/validate_submission.py predictions.csv
   ```
2. Encrypt it and create your submission folder:
   ```bash
   python scripts/encrypt_submission.py predictions.csv \
       --team my_team --model "GraphSAGE" --type human --notes "3 layers, hidden 128"
   ```
   `--type` is `human`, `llm` or `hybrid`.
3. Commit **only** `submissions/inbox/my_team/`, push to your fork, and open a Pull Request to `main`.
4. The scoring bot decrypts, scores and updates the leaderboard automatically, then comments on your PR.

### Required format
```csv
id,y_pred
0,0.02
1,0.91
```
One row per row of `test_nodes.csv` (use its `id` column). `y_pred` is the probability in [0, 1] that the
node is polyp; a node counts as foreground when `y_pred >= 0.5`.

### Scoring
Painting every foreground node's pixels gives a predicted mask per image. Dice and IoU are computed per
image (exactly, from node areas and `label_frac`) and averaged. An image with no polyp and no prediction scores 1.0.

PRs that change anything other than your own `submissions/inbox/<team>/` folder are rejected.

---

## 5. 🏆 Leaderboard

**Live:** https://ignatiusbalayo.github.io/PolypSegArena/leaderboard.html

- Ranked by mean Dice (Kaggle-style: ties share a rank)
- Search by team or model, filter by human / llm / hybrid
- Updated automatically after every scored submission

---

## 6. Getting Started

### Explore the Data
```python
import pandas as pd

nodes = pd.read_csv("data/public/train_nodes.csv")
edges = pd.read_csv("data/public/train_edges.csv")
test = pd.read_csv("data/public/test_nodes.csv")

g = nodes[nodes.image_id == "img_0001"]
print(len(g), "nodes;", (edges.image_id == "img_0001").sum(), "edges")
print("Test nodes to predict:", len(test))
```

### Build a Simple GNN
All graphs are stacked into one big disconnected graph (needs `pip install torch-geometric`).
```python
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch_geometric.nn import SAGEConv

feat = [c for c in nodes.columns if c not in ("image_id", "node_id", "label_frac", "label")]

def to_edge_index(n, e):
    row = pd.Series(range(len(n)), index=pd.MultiIndex.from_frame(n[["image_id", "node_id"]]))
    s = row.loc[pd.MultiIndex.from_frame(e[["image_id", "source"]])].to_numpy()
    t = row.loc[pd.MultiIndex.from_frame(e[["image_id", "target"]])].to_numpy()
    return torch.tensor(np.stack([np.r_[s, t], np.r_[t, s]]))      # undirected

x = torch.tensor(nodes[feat].values, dtype=torch.float)
mu, sd = x.mean(0), x.std(0) + 1e-6
x = (x - mu) / sd
edge_index = to_edge_index(nodes, edges)
y = torch.tensor(nodes["label_frac"].values, dtype=torch.float)

class Net(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.c1, self.c2, self.out = SAGEConv(d, h), SAGEConv(h, h), nn.Linear(h, 1)
    def forward(self, x, ei):
        return self.out(self.c2(self.c1(x, ei).relu(), ei).relu()).squeeze(-1)

model = Net(x.shape[1])
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(100):
    model.train()
    loss = nn.functional.binary_cross_entropy_with_logits(model(x, edge_index), y)
    opt.zero_grad(); loss.backward(); opt.step()
```

### Generate Predictions
```python
test_edges = pd.read_csv("data/public/test_edges.csv")
xt = (torch.tensor(test[feat].values, dtype=torch.float) - mu) / sd
model.eval()
with torch.no_grad():
    pred = torch.sigmoid(model(xt, to_edge_index(test, test_edges))).numpy()

pd.DataFrame({"id": test["id"], "y_pred": pred}).to_csv("predictions.csv", index=False)

# Validate before submitting!
# python competition/validate_submission.py predictions.csv
```

---

## 7. Rules

### Allowed
- Any graph neural network (GCN, GraphSAGE, GAT, GIN, graph transformers, ...), any GNN library, and publicly available pretrained weights
- Feature engineering on the provided node features, augmentation on graphs, loss design, post-processing
- Ensembles, unlimited offline training

### Not Allowed
- **Models that ignore the graph.** The model must use the edges (message passing or graph attention). A per-node MLP or gradient-boosting model on node features alone is not a valid entry
- Using the original Kvasir-SEG images or masks (including their test data) to train
- Looking up, downloading or matching against the original Kvasir-SEG dataset, or any other copy of it
- Manual labeling of test nodes
- Test-set peeking or probing the scorer
- **ONE submission per team** (enforced by the bot)

Graph-use is checked from the `--notes` / model description and by reviewers; it cannot be verified from predictions alone.

---

## 8. Baseline Performance

**Plain 3-layer GCN (hidden 64, dropout 0.3), full-batch, 300 epochs on CPU, class-weighted loss, decision offset tuned on 15% of the train graphs.**

| Metric | Value |
|:--|:--|
| mean Dice | 0.3755 |
| mean IoU | 0.2644 |
| CPU time | about 22.5 minutes (organizer's laptop CPU) |

Predicting every node as polyp scores 0.3021 Dice and perfect node labels reach 0.8845, so the baseline is a floor to beat, not a ceiling. Scores come from one run on one CPU and vary slightly with hardware.

---

## 9. Tips for Success
- Edge-aware models (GAT, GraphSAGE with mean/max pooling) can separate a polyp from similarly coloured mucosa better than a plain GCN.
- Test images are noisier than train: normalise features per graph or add feature noise during training.
- Train labels are imperfect; soft targets (`label_frac`) and robust losses help.
- Large polyps are under-represented in training: add graph-level context (e.g. global pooling) to the node features.

---

## 10. Resources
- Kipf & Welling, *Semi-Supervised Classification with GCNs*, ICLR 2017
- Hamilton et al., *GraphSAGE*, NeurIPS 2017; Velickovic et al., *GAT*, ICLR 2018
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io), [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)

---

## 11. References
- **Dataset:** Jha et al., *Kvasir-SEG: A Segmented Polyp Dataset*, MultiMedia Modeling (MMM) 2020.
- **Superpixels:** Achanta et al., *SLIC Superpixels*, IEEE TPAMI 2012.

---

## 12. License
Code: MIT License. Dataset: derived from Kvasir-SEG and subject to its original licence.

---

**Good luck!** 🚀
