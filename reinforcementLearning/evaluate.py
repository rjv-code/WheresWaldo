"""
evaluate.py — Compare Q-learning agent vs exhaustive baseline on test images.

Runs both strategies on the same images and prints a side-by-side comparison:
  - Baseline : checks every tile (original find_waldo.py behaviour)
  - Q-agent  : uses the trained Q-table, epsilon=0 (pure exploitation)

"""

import argparse
import time
from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from waldo_env import WaldoEnv
from q_agent import QAgent

REPO_ID     = "siddharthvader/waldo-yolov8-tiled"
WEIGHT_FILE = "waldo_yolov8s_tile512.pt"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model() -> YOLO:
    weights = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FILE)
    return YOLO(weights)


def run_baseline(env: WaldoEnv, image_path: str) -> dict:
    """Check every tile in fixed order — simulates the original script."""
    state = env.reset(image_path)
    n = env.n_tiles()

    # Image too small to tile — skip it
    if n == 0:
        return {"tiles_checked": 0, "total_tiles": 0, "found": False,
                "confidence": 0.0, "skipped": True}

    tiles_checked = 0
    found = False
    info  = {"confidence": 0.0, "found": False}

    for action in range(n):
        _, reward, done, info = env.step(action)
        tiles_checked += 1
        if done:
            found = info["found"]
            break

    return {
        "tiles_checked": tiles_checked,
        "total_tiles":   n,
        "found":         found,
        "confidence":    info["confidence"],
        "skipped":       False,
    }


def run_agent(env: WaldoEnv, agent: QAgent, image_path: str) -> dict:
    """Run the trained Q-agent with epsilon=0 (pure exploitation)."""
    original_eps  = agent.epsilon
    agent.epsilon = 0.0          # no exploration during evaluation

    state = env.reset(image_path)

    # Image too small to tile — skip it
    if env.n_tiles() == 0:
        agent.epsilon = original_eps
        return {"tiles_checked": 0, "total_tiles": 0, "found": False,
                "confidence": 0.0, "skipped": True}

    done = False
    info = {"confidence": 0.0, "found": False, "tiles_checked": 0}

    while not done:
        valid  = env.valid_actions()
        if not valid:
            break
        action = agent.select_action(state, valid)
        state, reward, done, info = env.step(action)

    agent.epsilon = original_eps  # restore

    return {
        "tiles_checked": info["tiles_checked"],
        "total_tiles":   env.n_tiles(),
        "found":         info["found"],
        "confidence":    info["confidence"],
    }


def evaluate(
    images:      list[str],
    model:       YOLO,
    agent:       QAgent,
    tile_size:   int   = 512,
    overlap:     float = 0.125,
    found_thresh:float = 0.3,
    conf:        float = 0.1,
) -> None:

    env = WaldoEnv(
        model,
        tile_size=tile_size,
        overlap=overlap,
        found_threshold=found_thresh,
        conf=conf,
    )

    baseline_tiles  = []
    agent_tiles     = []
    baseline_found  = 0
    agent_found     = 0

    header = (f"{'Image':<30}  {'Baseline tiles':>14}  {'Agent tiles':>12}  "
              f"{'Saved':>6}  {'B-found':>8}  {'A-found':>8}")
    print(header)
    print("-" * len(header))

    skipped = 0
    for image_path in images:
        name = Path(image_path).name

        b = run_baseline(env, image_path)
        a = run_agent(env, agent, image_path)

        # Skip images that are too small to tile
        if b.get("skipped") or a.get("skipped"):
            print(f"{name:<30}  SKIPPED (image smaller than tile size)")
            skipped += 1
            continue

        saved = b["tiles_checked"] - a["tiles_checked"]
        baseline_tiles.append(b["tiles_checked"])
        agent_tiles.append(a["tiles_checked"])
        if b["found"]: baseline_found += 1
        if a["found"]: agent_found    += 1

        print(
            f"{name:<30}  "
            f"{b['tiles_checked']:>6}/{b['total_tiles']:<6}  "
            f"{a['tiles_checked']:>5}/{a['total_tiles']:<6}  "
            f"{saved:>+6}  "
            f"{'✓' if b['found'] else '✗':>8}  "
            f"{'✓' if a['found'] else '✗':>8}"
        )

    n = len(images) - skipped
    print("\n" + "=" * len(header))
    print(f"  Images evaluated  : {n}  ({skipped} skipped — too small for tile size)")
    print(f"  Baseline  — found : {baseline_found}/{n}  "
          f"avg tiles: {sum(baseline_tiles)/n:.1f}")
    print(f"  Q-agent   — found : {agent_found}/{n}  "
          f"avg tiles: {sum(agent_tiles)/n:.1f}")
    if baseline_tiles:
        avg_saved = (sum(baseline_tiles) - sum(agent_tiles)) / n
        pct_saved = 100 * avg_saved / (sum(baseline_tiles) / n)
        print(f"  Avg tiles saved   : {avg_saved:.1f}  ({pct_saved:.1f}% reduction)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Q-agent vs exhaustive baseline."
    )
    parser.add_argument(
        "--images", required=True,
        help="Folder of test images (should not overlap with training images)"
    )
    parser.add_argument("--qtable",       default="qtable.pkl")
    parser.add_argument("--tile-size",    type=int,   default=512)
    parser.add_argument("--overlap",      type=float, default=0.125)
    parser.add_argument("--found-thresh", type=float, default=0.3)
    parser.add_argument("--conf",         type=float, default=0.1)
    args = parser.parse_args()

    images = [
        str(p) for p in Path(args.images).iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not images:
        print(f"No images found in {args.images}")
        return

    print(f"Loading model and Q-table ...")
    model = load_model()
    agent = QAgent(n_tiles=1)   # n_tiles overwritten by load()
    agent.load(args.qtable)

    print(f"\nEvaluating on {len(images)} image(s) ...\n")
    evaluate(
        images       = images,
        model        = model,
        agent        = agent,
        tile_size    = args.tile_size,
        overlap      = args.overlap,
        found_thresh = args.found_thresh,
        conf         = args.conf,
    )


if __name__ == "__main__":
    main()