"""ORGANIZER ONLY - build the PolypSegArena dataset from raw Kvasir-SEG.

Raw input (not committed):
    data/raw/Kvasir-SEG/images/*.jpg     1,000 colonoscopy images
    data/raw/Kvasir-SEG/masks/*.jpg      matching polyp masks

What this script does (all seeded, fully reproducible):
  1. Resize every image to 256x256 (masks: bilinear + threshold at 127 -> binary).
  2. Rename every image to img_0001... in shuffled order (hides original file names).
  3. Split into TEST (default 400, natural distribution) and a TRAIN pool.
  4. MODIFICATION 1 - size imbalance: in TRAIN, polyps whose area is in the largest
     quartile are kept with probability --large-keep, so small polyps dominate training
     while the test set still contains the natural share of large ones.
  5. MODIFICATION 2 - anti-matching geometry (masks transformed too): random crop-and-
     resize, horizontal / vertical flip and 90-degree rotation. Stronger on test than train.
     A crop must keep at least 90% of the polyp area, so test images never lose their polyp.
  6. MODIFICATION 3 - colour jitter: brightness, contrast, saturation and hue shifts.
  7. MODIFICATION 4 - noise: a fraction of test (50%) and train (20%) images get one of
     Gaussian noise, Gaussian blur or heavy JPEG compression.
  8. MODIFICATION 5 - label noise: --label-noise (10%) of TRAIN masks are randomly
     dilated or eroded by 2-5 px. Test masks are never perturbed.

Modifications 2-4 exist so that test images cannot easily be matched back to the public
Kvasir-SEG images (and their masks) by file name, hashing or simple feature matching.
They reduce leakage; they do not make it impossible.

Outputs (all PRIVATE and git-ignored - raw pixels are never published):
    data/private/staging/train/images/*.png    data/private/staging/train/masks/*.png (0/255)
    data/private/staging/test/images/*.png     data/private/staging/test/masks/*.png  (0/255)
    data/private/staging/{train,test}_index.csv
    data/private/modification_log.json         (source file + applied changes per image)
    data/private/prep_stats.json

Next step: python scripts/build_graphs.py   (turns the staged images into the public graph files)

Usage:
    python scripts/prepare_data.py --raw data/raw/Kvasir-SEG
"""

import argparse
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]

SIZE = 256

# Strength of the geometric / colour changes per split (test is stronger on purpose).
GEOM = {
    "train": {"crop_min": 0.85, "p_hflip": 0.5, "p_vflip": 0.25, "p_rot90": 0.25, "color": 0.15},
    "test": {"crop_min": 0.70, "p_hflip": 0.5, "p_vflip": 0.50, "p_rot90": 0.50, "color": 0.30},
}
MIN_MASK_KEPT = 0.90   # a crop must keep at least this share of the polyp area


def corrupt(img: Image.Image, rng: np.random.Generator, mild: bool) -> tuple[Image.Image, str]:
    kind = str(rng.choice(["gauss", "blur", "jpeg"]))
    if kind == "gauss":
        sigma = rng.uniform(8, 16) if mild else rng.uniform(15, 30)
        arr = np.asarray(img, dtype=np.float32) + rng.normal(0, sigma, (img.height, img.width, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif kind == "blur":
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 1.2) if mild else rng.uniform(1.5, 2.5)))
    else:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(rng.integers(30, 45) if mild else rng.integers(10, 25)))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img, kind


def color_jitter(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    """Brightness / contrast / saturation factors in [1-s, 1+s] and a hue shift of up to +-s*40 units."""
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(1 - strength, 1 + strength))
    h, s, v = img.convert("HSV").split()
    shift = int(rng.uniform(-strength, strength) * 40)
    h = Image.fromarray(((np.asarray(h).astype(np.int16) + shift) % 256).astype(np.uint8))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def geometric(img: Image.Image, mask: np.ndarray, rng: np.random.Generator, cfg: dict):
    """Random crop-and-resize, flips and 90-degree rotation applied to image AND mask.

    img: 256x256 RGB PIL image; mask: 256x256 bool array. Returns (img, mask, description).
    """
    desc = []
    total = float(mask.sum())
    mimg = Image.fromarray((mask * 255).astype(np.uint8))
    for _ in range(10):  # retry until the crop keeps the polyp
        k = np.sqrt(rng.uniform(cfg["crop_min"] ** 2, 1.0))
        c = max(8, int(round(SIZE * k)))
        x0, y0 = int(rng.integers(0, SIZE - c + 1)), int(rng.integers(0, SIZE - c + 1))
        if total == 0 or mask[y0:y0 + c, x0:x0 + c].sum() / total >= MIN_MASK_KEPT:
            box = (x0, y0, x0 + c, y0 + c)
            img = img.crop(box).resize((SIZE, SIZE), Image.BILINEAR)
            mimg = mimg.crop(box).resize((SIZE, SIZE), Image.NEAREST)
            desc.append(f"crop{k:.2f}")
            break
    for flag, op, name in (
        ("p_hflip", Image.FLIP_LEFT_RIGHT, "hflip"),
        ("p_vflip", Image.FLIP_TOP_BOTTOM, "vflip"),
        ("p_rot90", Image.ROTATE_90, "rot90"),
    ):
        if rng.random() < cfg[flag]:
            img, mimg = img.transpose(op), mimg.transpose(op)
            desc.append(name)
    return img, np.asarray(mimg) > 127, "+".join(desc) or "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/Kvasir-SEG"))
    ap.add_argument("--test-size", type=int, default=400)
    ap.add_argument("--large-keep", type=float, default=0.5)
    ap.add_argument("--noise-test", type=float, default=0.50)
    ap.add_argument("--noise-train", type=float, default=0.20)
    ap.add_argument("--label-noise", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = Path(args.raw)
    files = sorted(p.name for p in (raw / "images").glob("*.jpg"))
    if not files:
        sys.exit(f"No images found in {raw / 'images'}")
    files = [files[i] for i in rng.permutation(len(files))]
    print(f"{len(files)} raw images")

    items = []
    for name in files:
        img = Image.open(raw / "images" / name).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        m = Image.open(raw / "masks" / name).convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        items.append((name, img, np.asarray(m) > 127))

    test_items, train_pool = items[: args.test_size], items[args.test_size :]
    area_cut = np.quantile([m.mean() for _, _, m in train_pool], 0.75)  # on the ORIGINAL masks

    stage = ROOT / "data" / "private" / "staging"
    for sub in ("train/images", "train/masks", "test/images", "test/masks"):
        (stage / sub).mkdir(parents=True, exist_ok=True)

    mod_log, noise_counts, geom_counts = {}, Counter(), Counter()
    train_rows, test_rows = [], []
    next_id = 1

    def new_id() -> str:
        nonlocal next_id
        iid = f"img_{next_id:04d}"
        next_id += 1
        return iid

    def save_pair(split: str, iid: str, img: Image.Image, mask: np.ndarray) -> None:
        img.save(stage / split / "images" / f"{iid}.png")
        Image.fromarray((mask * 255).astype(np.uint8)).save(stage / split / "masks" / f"{iid}.png")

    # ---- test split (natural masks, transformed + noisy images)
    for name, img, mask in test_items:
        iid = new_id()
        img, mask, geom_desc = geometric(img, mask, rng, GEOM["test"])
        img = color_jitter(img, rng, GEOM["test"]["color"])
        kind = "none"
        if rng.random() < args.noise_test:
            img, kind = corrupt(img, rng, mild=False)
        save_pair("test", iid, img, mask)
        test_rows.append({"image_id": iid, "file_name": f"{iid}.png"})
        mod_log[iid] = {"split": "test", "source": name, "geometry": geom_desc, "noise": kind}
        noise_counts[("test", kind)] += 1
        geom_counts.update(p for p in geom_desc.split("+") if p != "none" and not p.startswith("crop"))

    # ---- train split (size imbalance + geometry + noise + label noise)
    dropped = 0
    for name, img, mask in train_pool:
        if mask.mean() > area_cut and rng.random() > args.large_keep:
            dropped += 1
            continue
        iid = new_id()
        img, mask, geom_desc = geometric(img, mask, rng, GEOM["train"])
        img = color_jitter(img, rng, GEOM["train"]["color"])
        kind = "none"
        if rng.random() < args.noise_train:
            img, kind = corrupt(img, rng, mild=True)
        label_pert = "none"
        if rng.random() < args.label_noise:
            it = int(rng.integers(2, 6))
            if rng.random() < 0.5:
                mask, label_pert = ndimage.binary_dilation(mask, iterations=it), f"dilate{it}"
            else:
                mask, label_pert = ndimage.binary_erosion(mask, iterations=it), f"erode{it}"
        save_pair("train", iid, img, mask)
        train_rows.append({"image_id": iid, "file_name": f"{iid}.png"})
        mod_log[iid] = {"split": "train", "source": name, "geometry": geom_desc, "noise": kind,
                        "label_noise": label_pert}
        noise_counts[("train", kind)] += 1

    pd.DataFrame(train_rows).to_csv(stage / "train_index.csv", index=False)
    pd.DataFrame(test_rows).to_csv(stage / "test_index.csv", index=False)
    (ROOT / "data" / "private" / "modification_log.json").write_text(json.dumps(mod_log, indent=2))

    def area(split: str, rows: list) -> list:
        return [np.asarray(Image.open(stage / split / "masks" / r["file_name"])).astype(bool).mean() for r in rows]

    tr_area, te_area = area("train", train_rows), area("test", test_rows)
    stats = {
        "seed": args.seed,
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "train_large_polyps_dropped": dropped,
        "train_median_polyp_area_fraction": round(float(np.median(tr_area)), 4),
        "test_median_polyp_area_fraction": round(float(np.median(te_area)), 4),
        "test_empty_masks": int(sum(1 for a in te_area if a == 0)),
        "noisy_images": {f"{s}:{k}": v for (s, k), v in sorted(noise_counts.items())},
        "test_flip_rotate_counts": dict(geom_counts),
    }
    (ROOT / "data" / "private" / "prep_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print("\nNext: python scripts/build_graphs.py")


if __name__ == "__main__":
    main()
