"""ORGANIZER ONLY - turn the staged images into the public GRAPH dataset.

Input (private, from scripts/prepare_data.py):
    data/private/staging/{train,test}/images/*.png, masks/*.png, {train,test}_index.csv

Each 256x256 image becomes ONE graph:
  * nodes   = SLIC superpixels (about --n-segments per image)
  * edges   = region adjacency (two superpixels that touch), undirected, listed once
  * features per node = 17 hand-crafted statistics (colour, texture, position, shape)
  * label   = share of the node's pixels that belong to the polyp (label_frac), plus
              label = 1 if label_frac > 0.5

NO raw pixels are published. Participants only receive these graphs.

Outputs (public, committed):
    data/public/train_nodes.csv     image_id,node_id,<17 features>,label_frac,label
    data/public/train_edges.csv     image_id,source,target
    data/public/test_nodes.csv      id,image_id,node_id,<17 features>      (no labels)
    data/public/test_edges.csv      image_id,source,target
    data/public/sample_submission.csv   id,y_pred
    data/public/dataset_stats.json
Output (private, git-ignored):
    data/private/test_labels.csv    id,image_id,node_id,area,frac
Then run: python scripts/encrypt_labels.py

Scoring (see competition/metrics.py): a node is predicted foreground if y_pred >= 0.5.
Painting node predictions back onto pixels gives an exact pixel-level Dice / IoU per image,
computed from the node areas and label_frac values - no segment maps are needed.

Usage:
    python scripts/build_graphs.py
    python scripts/build_graphs.py --n-segments 120 --compactness 10
Needs: pip install scikit-image
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.filters import sobel
    from skimage.segmentation import slic
except ImportError:
    sys.exit("scikit-image is required: pip install scikit-image")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "competition"))
from metrics import calculate_metrics  # noqa: E402

STAGE = ROOT / "data" / "private" / "staging"
PUB = ROOT / "data" / "public"
FEATURES = [
    "area", "r_mean", "g_mean", "b_mean", "r_std", "g_std", "b_std",
    "hue_cos", "hue_sin", "sat_mean", "val_mean", "grad_mean",
    "cx", "cy", "bbox_w", "bbox_h", "fill_ratio",
]


def build_graph(img: np.ndarray, mask: np.ndarray, n_segments: int, compactness: float):
    """img: HxWx3 float in [0,1]; mask: HxW bool. Returns (features[n,17], edges[m,2], frac[n], area[n])."""
    h, w, _ = img.shape
    seg = slic(img, n_segments=n_segments, compactness=compactness, sigma=1.0, start_label=0, channel_axis=-1)
    _, seg = np.unique(seg, return_inverse=True)
    seg = seg.reshape(h, w)
    n = int(seg.max()) + 1
    flat = seg.ravel()
    counts = np.bincount(flat, minlength=n).astype(np.float64)

    def mean_of(values: np.ndarray) -> np.ndarray:
        return np.bincount(flat, weights=values.ravel(), minlength=n) / counts

    hsv = rgb2hsv(img)
    grad = sobel(rgb2gray(img))
    yy, xx = np.mgrid[0:h, 0:w]
    feats = {"area": counts / (h * w)}
    for i, c in enumerate("rgb"):
        m = mean_of(img[..., i])
        feats[f"{c}_mean"] = m
        feats[f"{c}_std"] = np.sqrt(np.clip(mean_of(img[..., i] ** 2) - m ** 2, 0, None))
    feats["hue_cos"] = mean_of(np.cos(2 * np.pi * hsv[..., 0]))
    feats["hue_sin"] = mean_of(np.sin(2 * np.pi * hsv[..., 0]))
    feats["sat_mean"] = mean_of(hsv[..., 1])
    feats["val_mean"] = mean_of(hsv[..., 2])
    feats["grad_mean"] = mean_of(grad)
    feats["cx"] = mean_of(xx / (w - 1))
    feats["cy"] = mean_of(yy / (h - 1))
    # bounding box of every superpixel
    x_min, x_max = np.full(n, w), np.zeros(n)
    y_min, y_max = np.full(n, h), np.zeros(n)
    np.minimum.at(x_min, flat, xx.ravel())
    np.maximum.at(x_max, flat, xx.ravel())
    np.minimum.at(y_min, flat, yy.ravel())
    np.maximum.at(y_max, flat, yy.ravel())
    bw, bh = (x_max - x_min + 1), (y_max - y_min + 1)
    feats["bbox_w"], feats["bbox_h"] = bw / w, bh / h
    feats["fill_ratio"] = counts / (bw * bh)
    features = np.stack([feats[name] for name in FEATURES], axis=1)

    pairs = np.concatenate([
        np.stack([seg[:, :-1].ravel(), seg[:, 1:].ravel()], 1),
        np.stack([seg[:-1, :].ravel(), seg[1:, :].ravel()], 1),
    ])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)

    frac = np.bincount(flat, weights=mask.ravel().astype(np.float64), minlength=n) / counts
    return features, pairs, frac, counts


def load_split(split: str, args):
    index = pd.read_csv(STAGE / f"{split}_index.csv")
    node_rows, edge_rows, label_rows = [], [], []
    for k, (iid, fname) in enumerate(zip(index["image_id"], index["file_name"]), 1):
        img = np.asarray(Image.open(STAGE / split / "images" / fname).convert("RGB"), dtype=np.float64) / 255.0
        mask = np.asarray(Image.open(STAGE / split / "masks" / fname)) > 127
        feats, edges, frac, area = build_graph(img, mask, args.n_segments, args.compactness)
        n = len(feats)
        node_df = pd.DataFrame(np.round(feats, 5), columns=FEATURES)
        node_df.insert(0, "node_id", np.arange(n))
        node_df.insert(0, "image_id", iid)
        node_df["label_frac"] = np.round(frac, 4)
        node_df["label"] = (frac > 0.5).astype(int)
        node_rows.append(node_df)
        edge_rows.append(pd.DataFrame({"image_id": iid, "source": edges[:, 0], "target": edges[:, 1]}))
        label_rows.append(pd.DataFrame({"image_id": iid, "node_id": np.arange(n), "area": area.astype(int),
                                        "frac": np.round(frac, 4)}))
        if k % 100 == 0:
            print(f"  {split}: {k}/{len(index)} graphs")
    return pd.concat(node_rows, ignore_index=True), pd.concat(edge_rows, ignore_index=True), pd.concat(
        label_rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-segments", type=int, default=100)
    ap.add_argument("--compactness", type=float, default=10.0)
    args = ap.parse_args()

    if not (STAGE / "train_index.csv").exists():
        sys.exit("Run scripts/prepare_data.py first (staging folder not found).")
    PUB.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "private").mkdir(parents=True, exist_ok=True)

    tr_nodes, tr_edges, _ = load_split("train", args)
    te_nodes, te_edges, te_labels = load_split("test", args)

    tr_nodes.to_csv(PUB / "train_nodes.csv", index=False)
    tr_edges.to_csv(PUB / "train_edges.csv", index=False)
    te_nodes = te_nodes.drop(columns=["label_frac", "label"])
    te_nodes.insert(0, "id", np.arange(len(te_nodes)))
    te_nodes.to_csv(PUB / "test_nodes.csv", index=False)
    te_edges.to_csv(PUB / "test_edges.csv", index=False)
    te_labels.insert(0, "id", np.arange(len(te_labels)))
    te_labels.to_csv(ROOT / "data" / "private" / "test_labels.csv", index=False)
    pd.DataFrame({"id": te_nodes["id"], "y_pred": 0.0}).to_csv(PUB / "sample_submission.csv", index=False)

    oracle = calculate_metrics(
        pd.DataFrame({"id": te_labels["id"], "y_pred": (te_labels["frac"] > 0.5).astype(float)}), te_labels)
    all_fg = calculate_metrics(pd.DataFrame({"id": te_labels["id"], "y_pred": 1.0}), te_labels)
    stats = {
        "n_segments_target": args.n_segments,
        "compactness": args.compactness,
        "num_node_features": len(FEATURES),
        "train_graphs": int(tr_nodes["image_id"].nunique()),
        "test_graphs": int(te_nodes["image_id"].nunique()),
        "train_nodes": int(len(tr_nodes)),
        "test_nodes": int(len(te_nodes)),
        "train_edges": int(len(tr_edges)),
        "test_edges": int(len(te_edges)),
        "mean_nodes_per_graph": round(len(tr_nodes) / tr_nodes["image_id"].nunique(), 1),
        "train_positive_node_rate": round(float(tr_nodes["label"].mean()), 4),
        "test_positive_node_rate": round(float((te_labels["frac"] > 0.5).mean()), 4),
        "test_ceiling_mean_dice_perfect_node_labels": round(oracle["mean_dice"], 4),
        "test_mean_dice_all_foreground": round(all_fg["mean_dice"], 4),
    }
    (PUB / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("\nNext: python scripts/generate_keys.py (first time only), then python scripts/encrypt_labels.py")


if __name__ == "__main__":
    main()
