
import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

REPO_ID = "siddharthvader/waldo-yolov8-tiled"

WEIGHT_FILES = {
    256: "waldo_yolov8s_tile256.pt",   # 91.4% detection rate — maximum accuracy
    384: "waldo_yolov8s_tile384.pt",   # 79.3%
    512: "waldo_yolov8s_tile512.pt",   # 87.9% —  (best balance)
    640: "waldo_yolov8s_tile640.pt",   # 77.6%
    768: "waldo_yolov8s_tile768.pt",   # 79.3% — maximum speed
}


def load_model(tile_size: int) -> YOLO:
    filename = WEIGHT_FILES[tile_size]
    print(f"[1/3] Downloading weights: {filename} ...")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    print(f"      Cached at: {weights_path}")
    model = YOLO(weights_path)
    print("      Model loaded.\n")
    return model


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def iou(box1: dict, box2: dict) -> float:
    ix1 = max(box1["x1"], box2["x1"])
    iy1 = max(box1["y1"], box2["y1"])
    ix2 = min(box1["x2"], box2["x2"])
    iy2 = min(box1["y2"], box2["y2"])

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Tiled inference
# ---------------------------------------------------------------------------

def find_waldo(
    image_path: str,
    model: YOLO,
    tile_size: int = 512,
    overlap: float = 0.125,
    conf: float = 0.1,
) -> list[dict]:
 
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    print(f"[2/3] Running tiled inference on {img_w}×{img_h}px image ...")
    print(f"      tile_size={tile_size}px  overlap={overlap*100:.1f}%  conf={conf}")

    stride = int(tile_size * (1 - overlap))
    all_detections: list[dict] = []
    tile_count = 0

    for y in range(0, img_h - tile_size + 1, stride):
        for x in range(0, img_w - tile_size + 1, stride):
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            results = model.predict(tile, conf=conf, verbose=False, imgsz=tile_size)
            tile_count += 1

            for result in results:
                for box in result.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    all_detections.append(
                        {
                            "x1": float(x + bx1),
                            "y1": float(y + by1),
                            "x2": float(x + bx2),
                            "y2": float(y + by2),
                            "confidence": float(box.conf[0]),
                        }
                    )

    print(f"      Scanned {tile_count} tiles, found {len(all_detections)} raw detections.")

    # Non-maximum suppression — greedy, IoU threshold 0.5
    all_detections.sort(key=lambda d: d["confidence"], reverse=True)
    keep: list[dict] = []
    while all_detections:
        best = all_detections.pop(0)
        keep.append(best)
        all_detections = [d for d in all_detections if iou(best, d) < 0.5]

    print(f"      After NMS: {len(keep)} detection(s).\n")
    return keep


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_detections(image_path: str, detections: list[dict], output_path: str) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to load a reasonable font; fall back to PIL default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=24)
    except OSError:
        font = ImageFont.load_default()

    colors = ["#FF3B30", "#FF9500", "#34C759", "#007AFF", "#AF52DE"]

    for i, det in enumerate(detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["confidence"]

        # Bounding box (thick outline)
        line_w = max(3, int(min(img.width, img.height) / 300))
        for offset in range(line_w):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # Label background + text
        label = f"Waldo #{i+1}  {conf:.1%}"
        bbox = draw.textbbox((x1, y1), label, font=font)
        label_h = bbox[3] - bbox[1] + 6
        label_w = bbox[2] - bbox[0] + 8
        draw.rectangle([x1, y1 - label_h, x1 + label_w, y1], fill=color)
        draw.text((x1 + 4, y1 - label_h + 3), label, fill="white", font=font)

    img.save(output_path)
    print(f"[3/3] Saved annotated image → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Find Waldo using tiled YOLOv8 inference.")
    parser.add_argument("image", help="Path to the Where's Waldo image")
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        choices=list(WEIGHT_FILES.keys()),
        help="Tile size in pixels — must match a trained model (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.125,
        help="Tile overlap fraction 0.0–0.5 (default: 0.125 = 12.5%%)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Confidence threshold for detections (default: 0.1)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load model
    model = load_model(args.tile_size)

    # Run inference
    detections = find_waldo(
        str(image_path),
        model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf=args.conf,
    )

    # Report results
    if not detections:
        print("No Waldo found. Try lowering --conf (e.g. --conf 0.05).")
        sys.exit(0)

    print(f"Results ({len(detections)} detection(s), best first):")
    for i, det in enumerate(detections):
        cx = (det["x1"] + det["x2"]) / 2
        cy = (det["y1"] + det["y2"]) / 2
        print(
            f"  #{i+1}  confidence={det['confidence']:.1%}"
            f"  center=({cx:.0f}, {cy:.0f})"
            f"  box=({det['x1']:.0f},{det['y1']:.0f})→({det['x2']:.0f},{det['y2']:.0f})"
        )

    # Save annotated image
    output_path = image_path.stem + "_waldo_found" + image_path.suffix
    draw_detections(str(image_path), detections, output_path)


if __name__ == "__main__":
    main()
