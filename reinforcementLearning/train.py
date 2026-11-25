
import argparse
import random
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
    print("Loading YOLO model ...")
    weights = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FILE)
    model   = YOLO(weights)
    print("  Model ready.\n")
    return model


def collect_images(folder: str) -> list[str]:
    paths = [
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ]
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    print(f"Found {len(paths)} training image(s) in {folder}\n")
    return paths


def train(
    images:        list[str],
    model:         YOLO,
    n_episodes:    int   = 500,
    tile_size:     int   = 512,
    overlap:       float = 0.125,
    found_thresh:  float = 0.3,
    conf:          float = 0.1,
    save_every:    int   = 50,
    qtable_path:   str   = "qtable.pkl",
) -> QAgent:

    env = WaldoEnv(
        model,
        tile_size=tile_size,
        overlap=overlap,
        found_threshold=found_thresh,
        conf=conf,
    )

    # We don't know n_tiles until the first reset, so do a dry run
    state = env.reset(images[0])
    n_tiles = env.n_tiles()
    print(f"Grid size: {n_tiles} tiles per image\n")

    agent = QAgent(n_tiles=n_tiles)

    # Tracking metrics
    episode_rewards   = []
    tiles_to_find     = []   # tiles checked when Waldo found (success only)
    success_count     = 0

    print(f"{'Episode':>8}  {'Image':<30}  {'Reward':>7}  "
          f"{'Tiles':>6}  {'Found':>6}  {'ε':>6}  {'Q entries':>10}")
    print("-" * 82)

    t0 = time.time()

    for episode in range(1, n_episodes + 1):
        image_path = random.choice(images)

        # --- Reset environment ---
        state = env.reset(image_path)
        total_reward = 0.0
        done         = False

        while not done:
            valid = env.valid_actions()
            action = agent.select_action(state, valid)

            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions()

            agent.update(state, action, reward, next_state, next_valid)

            state         = next_state
            total_reward += reward

        # End of episode
        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        found = info["found"] or any(
            # check if any tile in final state is FOUND (status 2)
            s == 2 for s in state
        )
        if found:
            success_count += 1
            tiles_to_find.append(info["tiles_checked"])

        #  Logging 
        if episode % 10 == 0 or episode == 1:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            avg_tiles  = (
                sum(tiles_to_find[-10:]) / min(10, len(tiles_to_find))
                if tiles_to_find else float("nan")
            )
            stats = agent.stats()
            print(
                f"{episode:>8}  {Path(image_path).name:<30}  "
                f"{avg_reward:>7.2f}  {avg_tiles:>6.1f}  "
                f"{success_count:>6}  {stats['epsilon']:>6.3f}  "
                f"{stats['q_entries']:>10}"
            )

        #  Periodic save 
        if episode % save_every == 0:
            agent.save(qtable_path)

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Success rate : {success_count}/{n_episodes} "
          f"({100*success_count/n_episodes:.1f}%)")
    if tiles_to_find:
        print(f"  Avg tiles to find Waldo : {sum(tiles_to_find)/len(tiles_to_find):.2f} "
              f"(baseline: {n_tiles})")

    agent.save(qtable_path)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Q-learning agent for Waldo tile selection."
    )
    parser.add_argument(
        "--images", required=True,
        help="Folder containing Where's Waldo training images"
    )
    parser.add_argument("--episodes",    type=int,   default=500)
    parser.add_argument("--tile-size",   type=int,   default=512)
    parser.add_argument("--overlap",     type=float, default=0.125)
    parser.add_argument("--found-thresh",type=float, default=0.3,
                        help="YOLO confidence >= this counts as Waldo found")
    parser.add_argument("--conf",        type=float, default=0.1,
                        help="YOLO detection floor confidence")
    parser.add_argument("--save-every",  type=int,   default=50,
                        help="Save Q-table every N episodes")
    parser.add_argument("--qtable",      default="qtable.pkl",
                        help="Output path for the Q-table")
    args = parser.parse_args()

    model  = load_model()
    images = collect_images(args.images)

    train(
        images        = images,
        model         = model,
        n_episodes    = args.episodes,
        tile_size     = args.tile_size,
        overlap       = args.overlap,
        found_thresh  = args.found_thresh,
        conf          = args.conf,
        save_every    = args.save_every,
        qtable_path   = args.qtable,
    )


if __name__ == "__main__":
    main()
