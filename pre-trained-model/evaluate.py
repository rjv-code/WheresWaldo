"""
Accuracy evaluation for a YOLOv8 Waldo detector on a labeled YOLO-format split.

The split folder must contain images/ and labels/ (the layout Roboflow exports, e.g. test/).

Examples
--------
# Pretrained tiled model (author's method), 256px tiles
python evaluate.py --weights waldo_yolov8s_tile256.pt --data path/to/test --tile 256

# Same, but also rescale each image first (tries several widths, keeps the best box)
python evaluate.py --weights waldo_yolov8s_tile256.pt --data path/to/test --tile 256 --widths 160 256 384

# A model fine-tuned on whole 640px images (no tiling)
python evaluate.py --weights best.pt --data path/to/test --tile 0

Metrics
-------
Detection rate : images containing Waldo where the top-confidence box overlaps a true box (IoU >= --iou)
Top-3 rate     : same, but any of the 3 most confident boxes may match
Avg IoU (hits) : how tightly the box fits Waldo when it is found
False alarms   : images with no Waldo where the model still reports a box with confidence >= --fa-conf
A per-image CSV is written so you can see exactly which images failed.
"""
import argparse
import csv
import glob
import os

from PIL import Image
from ultralytics import YOLO

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def tile_starts(length, tile, stride):
    """Tile start positions that also cover the right/bottom edge."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets, thr=0.5):
    """dets: [x1, y1, x2, y2, conf]. Keeps the most confident box among overlapping ones."""
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best, d) < thr]
    return keep


def detect(model, img, tile, overlap, conf):
    """Run the model on a PIL image, whole (tile=0) or in overlapping tiles."""
    dets = []
    if tile == 0:
        r = model.predict(img, conf=conf, verbose=False)[0]
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            dets.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c)])
        return nms(dets)

    W, H = img.size
    stride = max(1, int(tile * (1 - overlap)))
    for y in tile_starts(H, tile, stride):
        for x in tile_starts(W, tile, stride):
            crop = img.crop((x, y, x + tile, y + tile))  # pads with black past the edge
            r = model.predict(crop, imgsz=tile, conf=conf, verbose=False)[0]
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                dets.append([float(b[0]) + x, float(b[1]) + y, float(b[2]) + x, float(b[3]) + y, float(c)])
    return nms(dets)


def detect_multiscale(model, img, widths, tile, overlap, conf):
    """Resize to each width, detect, map boxes back to original coordinates."""
    if not widths:
        return detect(model, img, tile, overlap, conf)
    W, H = img.size
    all_dets = []
    for w in widths:
        s = w / W
        resized = img.resize((w, max(1, int(H * s))), Image.LANCZOS)
        for d in detect(model, resized, tile, overlap, conf):
            all_dets.append([d[0] / s, d[1] / s, d[2] / s, d[3] / s, d[4]])
    return nms(all_dets)


def load_gt(img_path, W, H):
    """Read YOLO labels (class cx cy w h, normalized) for an image -> pixel boxes."""
    split_dir = os.path.dirname(os.path.dirname(img_path))
    stem = os.path.splitext(os.path.basename(img_path))[0]
    lbl = os.path.join(split_dir, "labels", stem + ".txt")
    boxes = []
    if os.path.exists(lbl):
        with open(lbl) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return boxes


def main():
    ap = argparse.ArgumentParser(description="Evaluate a YOLOv8 Waldo detector.")
    ap.add_argument("--weights", required=True, help="path to .pt weights")
    ap.add_argument("--data", required=True, help="split folder containing images/ and labels/")
    ap.add_argument("--tile", type=int, default=256, help="tile size; 0 = run on the whole image")
    ap.add_argument("--overlap", type=float, default=0.25, help="tile overlap (0-0.5)")
    ap.add_argument("--conf", type=float, default=0.1, help="minimum confidence to keep a box")
    ap.add_argument("--widths", type=int, nargs="*", default=[], help="optional resize widths, e.g. 160 256 384")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to count as finding Waldo")
    ap.add_argument("--fa-conf", type=float, default=0.25, help="confidence that counts as a false alarm")
    ap.add_argument("--csv", default="eval_results.csv", help="per-image results file")
    args = ap.parse_args()

    images = sorted(p for p in glob.glob(os.path.join(args.data, "images", "*")) if p.lower().endswith(IMG_EXT))
    if not images:
        raise SystemExit(f"No images found in {os.path.join(args.data, 'images')}")

    model = YOLO(args.weights)
    rows = []
    for i, path in enumerate(images, 1):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        gts = load_gt(path, W, H)
        dets = detect_multiscale(model, img, args.widths, args.tile, args.overlap, args.conf)

        top_iou = max((iou(dets[0][:4], g) for g in gts), default=0.0) if dets else 0.0
        top3 = any(iou(d[:4], g) >= args.iou for d in dets[:3] for g in gts)
        rows.append({
            "file": os.path.basename(path),
            "has_waldo": bool(gts),
            "n_dets": len(dets),
            "top_conf": round(dets[0][4], 4) if dets else 0.0,
            "top_iou": round(top_iou, 4),
            "hit": bool(gts) and top_iou >= args.iou,
            "hit_top3": bool(gts) and top3,
            "false_alarm": (not gts) and any(d[4] >= args.fa_conf for d in dets),
        })
        print(f"\r{i}/{len(images)} images", end="", flush=True)
    print()

    pos = [r for r in rows if r["has_waldo"]]
    neg = [r for r in rows if not r["has_waldo"]]
    hits = [r for r in pos if r["hit"]]

    print(f"\nWeights: {args.weights}  | tile={args.tile}  widths={args.widths or 'native'}")
    print(f"Images: {len(rows)} ({len(pos)} with Waldo, {len(neg)} without)")
    if pos:
        print(f"Detection rate (top box):  {len(hits)}/{len(pos)} = {len(hits) / len(pos):.1%}")
        print(f"Detection rate (top 3):    {sum(r['hit_top3'] for r in pos)}/{len(pos)}")
        if hits:
            print(f"Avg IoU when found:        {sum(r['top_iou'] for r in hits) / len(hits):.3f}")
    if neg:
        fa = sum(r["false_alarm"] for r in neg)
        print(f"False alarms:              {fa}/{len(neg)} = {fa / len(neg):.1%}")
    if len(pos) < 30:
        print(f"\nNote: only {len(pos)} Waldo images, so each image moves the rate by "
              f"{100 / max(len(pos), 1):.0f} points. Treat small differences with caution.")

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-image results: {args.csv}")


if __name__ == "__main__":
    main()