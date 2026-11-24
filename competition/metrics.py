"""Dice / IoU for superpixel-graph segmentation.

Private labels (one row per test node): id,image_id,node_id,area,frac
  area = number of pixels in the superpixel, frac = share of those pixels that are polyp.

A node predicted foreground paints all its pixels as polyp, so per image
  TP = sum(area*frac over predicted nodes), pred = sum(area over predicted nodes),
  gt = sum(area*frac over all nodes)  ->  Dice = 2TP/(pred+gt), IoU = TP/(pred+gt-TP).
This equals the exact pixel-level metric of the painted mask. Images with an empty
ground truth and empty prediction score 1.0.
"""

import numpy as np
import pandas as pd


def calculate_metrics(preds: pd.DataFrame, labels: pd.DataFrame, threshold: float = 0.5) -> dict:
    df = labels.merge(preds[["id", "y_pred"]], on="id", how="left")
    if df["y_pred"].isna().any():
        raise ValueError("Some test nodes have no prediction.")
    df["on"] = (df["y_pred"].astype(float) >= threshold).astype(float)
    df["gt"] = df["area"] * df["frac"]
    df["tp"] = df["gt"] * df["on"]
    df["pr"] = df["area"] * df["on"]
    g = df.groupby("image_id")[["tp", "pr", "gt"]].sum()
    denom = g["pr"] + g["gt"]
    dice = np.where(denom > 0, 2 * g["tp"] / denom.where(denom > 0, 1), 1.0)
    union = g["pr"] + g["gt"] - g["tp"]
    iou = np.where(union > 0, g["tp"] / union.where(union > 0, 1), 1.0)
    return {"mean_dice": float(np.mean(dice)), "mean_iou": float(np.mean(iou))}
