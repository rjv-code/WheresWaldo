"""
RL-guided tile search for Waldo (Idea #1, version B): tabular Q-learning with stripe hints.

The image is split into a grid of overlapping cells (default 4x4). An agent starts on the
"stripiest" cell (most red-and-white pixels, like Waldo's shirt) and walks up/down/left/right.
Stepping onto an unchecked cell "checks" it with the YOLO model. If the model is confident
enough there, the agent declares Waldo found at that box and the episode ends.

  State   : current cell + a hint for each of the 4 neighbours (low / high / checked-or-edge)
            -> 16 * 3^4 = 1,296 states for a 4x4 grid, small enough for a Q-table
  Actions : 0 up, 1 down, 2 left, 3 right
  Reward  : -1 per move, +10 if the declared box is really Waldo (IoU >= 0.5 with the label),
            -5 for declaring a wrong box, -5 if it runs out of steps or cells

YOLO runs only ONCE per image (cached to a JSON file), so training takes seconds.

Usage (from pre-trained-model/, with the dataset unzipped into waldo-data/):
    python rl_tile_search.py --weights waldo_yolov8s_tile256.pt --data waldo-data

Trains on train/, evaluates on valid/ + test/, and compares the agent with:
    row-by-row scan, random order, highest-stripe-first (greedy, no RL),
    and a full scan that takes the most confident box (= evaluate.py's "top box").
"""
import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
ACTION_NAMES = ["up", "down", "left", "right"]


# ----------------------------------------------------------------------------- grid + hints
def cell_rects(W, H, g, ov):
    """g x g cells, each grown by `ov` of its size on every side so Waldo isn't cut at a border."""
    w, h = W / g, H / g
    rects = []
    for r in range(g):
        for c in range(g):
            x0, y0 = c * w, r * h
            rects.append((max(0.0, x0 - ov * w), max(0.0, y0 - ov * h),
                          min(float(W), x0 + w + ov * w), min(float(H), y0 + h + ov * h)))
    return rects


def stripe_scores(img, rects, work_w=900):
    """Fraction of pixels in each cell that are red with white just above/below (Waldo's shirt)."""
    W, H = img.size
    s = min(1.0, work_w / W)
    small = img.resize((max(1, int(W * s)), max(1, int(H * s)))) if s < 1 else img
    a = np.asarray(small.convert("RGB")).astype(np.int16)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    red = (R > 140) & (G < 100) & (B < 100)
    white = (R > 200) & (G > 200) & (B > 200)
    stripe = np.zeros_like(red)
    for k in (1, 2, 3):
        stripe[:-k] |= red[:-k] & white[k:]
        stripe[k:] |= red[k:] & white[:-k]
    out = []
    for x1, y1, x2, y2 in rects:
        xs1, ys1 = int(x1 * s), int(y1 * s)
        xs2, ys2 = max(xs1 + 1, int(x2 * s)), max(ys1 + 1, int(y2 * s))
        patch = stripe[ys1:ys2, xs1:xs2]
        out.append(float(patch.mean()) if patch.size else 0.0)
    return out


# ----------------------------------------------------------------------------- YOLO cache
def build_cache(args):
    """Run tiled YOLO once per image and save detections, labels and stripe scores."""
    config = {"weights": os.path.basename(args.weights), "grid": args.grid,
              "cell_overlap": args.cell_overlap, "tile": args.tile}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cached = json.load(f)
        if cached.get("config") == config:
            print(f"Using cached YOLO results from {args.cache}")
            return cached["entries"]
        print("Cache settings changed, rebuilding cache...")

    from ultralytics import YOLO
    from evaluate import detect, load_gt  # reuse the tested functions from evaluate.py

    model = YOLO(args.weights)
    entries = []
    for split in ("train", "valid", "test"):
        paths = sorted(p for p in glob.glob(os.path.join(args.data, split, "images", "*"))
                       if p.lower().endswith(IMG_EXT))
        for i, path in enumerate(paths, 1):
            img = Image.open(path).convert("RGB")
            W, H = img.size
            dets = detect(model, img, args.tile, 0.25, 0.05)  # low conf: keep candidates, threshold later
            entries.append({
                "split": split, "file": os.path.basename(path), "W": W, "H": H,
                "gts": load_gt(path, W, H), "dets": dets,
                "stripe": stripe_scores(img, cell_rects(W, H, args.grid, args.cell_overlap)),
            })
            print(f"\rCaching YOLO results [{split}] {i}/{len(paths)}", end="", flush=True)
        if paths:
            print()
    if not entries:
        raise SystemExit(f"No images found under {args.data}/(train|valid|test)/images")
    with open(args.cache, "w") as f:
        json.dump({"config": config, "entries": entries}, f)
    print(f"Saved {len(entries)} images to {args.cache}")
    return entries


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def cell_arrays(e, g, ov):
    """Per cell: confidence of its best detection, and whether that detection is really Waldo."""
    rects = cell_rects(e["W"], e["H"], g, ov)
    conf = np.zeros(g * g)
    hit = np.zeros(g * g, dtype=bool)
    for ci, (x1, y1, x2, y2) in enumerate(rects):
        best = None
        for d in e["dets"]:
            cx, cy = (d[0] + d[2]) / 2, (d[1] + d[3]) / 2
            if x1 <= cx < x2 and y1 <= cy < y2 and (best is None or d[4] > best[4]):
                best = d
        if best is not None:
            conf[ci] = best[4]
            hit[ci] = any(iou(best[:4], gt) >= 0.5 for gt in e["gts"])
    return conf, hit, np.array(e["stripe"], dtype=float)


def dihedral(arrs, g):
    """All 8 rotations/flips of the grid: 8x more training episodes from the same images."""
    grids = [a.reshape(g, g) for a in arrs]
    for k in range(4):
        for flip in (False, True):
            out = []
            for a in grids:
                t = np.rot90(a, k)
                out.append((np.fliplr(t) if flip else t).reshape(-1).copy())
            yield tuple(out)


# ----------------------------------------------------------------------------- environment
class TileSearchEnv:
    def __init__(self, g, stop_conf, top_k, max_steps):
        self.g, self.n = g, g * g
        self.stop_conf, self.top_k, self.max_steps = stop_conf, top_k, max_steps
        self.n_states = self.n * 81

    def reset(self, conf, hit, stripe):
        self.conf, self.hit = conf, hit
        order = np.argsort(-stripe, kind="stable")
        self.high = np.zeros(self.n, dtype=bool)
        for c in order[:self.top_k]:
            self.high[c] = stripe[c] > 0
        self.checked = np.zeros(self.n, dtype=bool)
        self.pos, self.steps, self.checks = int(order[0]), 0, 0
        self.result = None
        done, _ = self._check(self.pos)  # starting cell is checked for free
        return self.state(), done

    def _check(self, c):
        if self.checked[c]:
            return False, 0.0
        self.checked[c] = True
        self.checks += 1
        if self.conf[c] >= self.stop_conf:
            self.result = "found" if self.hit[c] else "false"
            return True, (10.0 if self.hit[c] else -5.0)
        if self.checked.all():
            self.result = "miss"
            return True, -5.0
        return False, 0.0

    def state(self):
        r, c = divmod(self.pos, self.g)
        code = 0
        for i, (dr, dc) in enumerate(MOVES):
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.g and 0 <= cc < self.g and not self.checked[rr * self.g + cc]:
                h = 1 if self.high[rr * self.g + cc] else 0
            else:
                h = 2  # already checked, or off the edge
            code += h * 3 ** i
        return self.pos * 81 + code

    def step(self, a):
        dr, dc = MOVES[a]
        r, c = divmod(self.pos, self.g)
        rr, cc = r + dr, c + dc
        self.steps += 1
        reward, done = -1.0, False
        if 0 <= rr < self.g and 0 <= cc < self.g:
            self.pos = rr * self.g + cc
            done, bonus = self._check(self.pos)
            reward += bonus
        if not done and self.steps >= self.max_steps:
            self.result, done = "miss", True
            reward -= 5.0
        return self.state(), reward, done


def greedy(q_row, rng):
    best = np.flatnonzero(q_row == q_row.max())
    return int(rng.choice(best))


def train_q(env, variants, episodes, alpha, gamma, eps_end, seed):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, 4))
    decay = max(1, int(episodes * 0.8))
    returns = []
    for ep in range(episodes):
        eps = max(eps_end, 1.0 - (1.0 - eps_end) * ep / decay)
        s, done = env.reset(*variants[rng.integers(len(variants))])
        total = 0.0
        while not done:
            a = int(rng.integers(4)) if rng.random() < eps else greedy(Q[s], rng)
            s2, r, done = env.step(a)
            target = r if done else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s, total = s2, total + r
        returns.append(total)
        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  episode {ep + 1}/{episodes}: avg return (last 1000) {np.mean(returns[-1000:]):.2f}")
    return Q


# ----------------------------------------------------------------------------- evaluation
def run_order(order, conf, hit, stop_conf):
    """Check cells in a fixed order; stop at the first confident one."""
    for k, c in enumerate(order, 1):
        if conf[c] >= stop_conf:
            return ("found" if hit[c] else "false"), k
    return "miss", len(order)


def run_agent(env, Q, conf, hit, stripe, rng):
    s, done = env.reset(conf, hit, stripe)
    while not done:
        s, _, done = env.step(greedy(Q[s], rng))
    return env.result, env.checks


def summarize(name, results, n_pos):
    """results: list of (has_waldo, outcome, cells_checked)."""
    pos = [r for r in results if r[0]]
    found = sum(r[1] == "found" for r in pos)
    false = sum(r[1] == "false" for r in results)
    avg_checks = np.mean([r[2] for r in results]) if results else 0.0
    return {"method": name, "found": found, "n_pos": n_pos,
            "found_rate": found / n_pos if n_pos else 0.0,
            "false_finds": false, "avg_cells_checked": float(avg_checks)}


def evaluate_methods(env, Q, data, stop_conf, seed):
    rng = np.random.default_rng(seed)
    n = env.n
    n_pos = sum(has for has, *_ in data)
    rows = {"Row-by-row scan": [], "Random order (avg of 20)": [], "Highest stripe first (no RL)": [],
            "Q-learning agent (B)": [], "Full scan, best box": []}
    rand_found = []
    for has, conf, hit, stripe in data:
        rows["Row-by-row scan"].append((has, *run_order(range(n), conf, hit, stop_conf)))
        rows["Highest stripe first (no RL)"].append(
            (has, *run_order(np.argsort(-stripe, kind="stable"), conf, hit, stop_conf)))
        rows["Q-learning agent (B)"].append((has, *run_agent(env, Q, conf, hit, stripe, rng)))
        best = int(np.argmax(conf))
        rows["Full scan, best box"].append((has, "found" if hit[best] and conf[best] > 0 else
                                            ("false" if conf[best] > 0 else "miss"), n))
        trials = [run_order(rng.permutation(n), conf, hit, stop_conf) for _ in range(20)]
        rand_found.append(trials)
    # random order: average over the 20 shuffles
    per_trial = [[(data[i][0], *rand_found[i][t]) for i in range(len(data))] for t in range(20)]
    summaries = [summarize("x", tr, n_pos) for tr in per_trial]
    out = []
    for name, res in rows.items():
        if name.startswith("Random"):
            out.append({"method": name, "found": float(np.mean([s["found"] for s in summaries])),
                        "n_pos": n_pos,
                        "found_rate": float(np.mean([s["found_rate"] for s in summaries])),
                        "false_finds": float(np.mean([s["false_finds"] for s in summaries])),
                        "avg_cells_checked": float(np.mean([s["avg_cells_checked"] for s in summaries]))})
        else:
            out.append(summarize(name, res, n_pos))
    return out


def pick_stop_conf(train_data, n, seed):
    """Choose the 'I found him' confidence on the TRAIN split only, using random check orders."""
    rng = np.random.default_rng(seed)
    best_t, best_rate = 0.3, -1.0
    for t in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7):
        found = 0
        for has, conf, hit, _ in train_data:
            for _ in range(10):
                found += run_order(rng.permutation(n), conf, hit, t)[0] == "found"
        if found > best_rate:
            best_t, best_rate = t, found
    return best_t


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'Method':32s} {'Found':>12s} {'False finds':>12s} {'Avg cells checked':>18s}")
    for r in rows:
        found = f"{r['found']:.1f}" if isinstance(r["found"], float) else str(r["found"])
        print(f"{r['method']:32s} {found + '/' + str(r['n_pos']):>8s} ({r['found_rate']:4.0%})"
              f" {r['false_finds']:>10.1f}  {r['avg_cells_checked']:>16.2f}")


def run_experiment(entries, args):
    g, ov = args.grid, args.cell_overlap
    prep = {"train": [], "eval": []}
    for e in entries:
        conf, hit, stripe = cell_arrays(e, g, ov)
        prep["train" if e["split"] == "train" else "eval"].append((bool(e["gts"]), conf, hit, stripe))
    if not prep["train"] or not prep["eval"]:
        raise SystemExit("Need images in train/ and in valid/ or test/.")

    stop_conf = args.stop_conf if args.stop_conf is not None else pick_stop_conf(prep["train"], g * g, args.seed)
    print(f"\nStop confidence: {stop_conf}  | grid {g}x{g}  | train {len(prep['train'])} images, "
          f"eval {len(prep['eval'])} images")
    reachable = sum(hit.any() for has, _, hit, _ in prep["eval"] if has)
    print(f"Eval images where YOLO has Waldo in some cell (best any method can do): "
          f"{reachable}/{sum(h for h, *_ in prep['eval'])}")

    env = TileSearchEnv(g, stop_conf, args.top_k, args.max_steps)
    variants = [v for _, conf, hit, stripe in prep["train"] for v in dihedral((conf, hit, stripe), g)]
    print(f"Training Q-learning on {len(variants)} grid variants (8 flips/rotations per image)...")
    Q = train_q(env, variants, args.episodes, args.alpha, args.gamma, args.eps_end, args.seed)
    np.save(args.q_out, Q)
    visited = int((np.abs(Q).sum(axis=1) > 0).sum())
    print(f"Q-table: {visited}/{env.n_states} states visited, saved to {args.q_out}")

    train_rows = evaluate_methods(env, Q, prep["train"], stop_conf, args.seed)
    eval_rows = evaluate_methods(env, Q, prep["eval"], stop_conf, args.seed)
    print_table("TRAIN split (sanity check, agent has seen these):", train_rows)
    print_table("VALID + TEST (unseen by the agent) - report these:", eval_rows)
    with open(args.results_out, "w") as f:
        json.dump({"stop_conf": stop_conf, "args": vars(args), "train": train_rows, "eval": eval_rows}, f, indent=2)
    print(f"\nResults saved to {args.results_out}")
    return eval_rows


def main():
    ap = argparse.ArgumentParser(description="Q-learning tile search for Waldo (version B, with stripe hints).")
    ap.add_argument("--weights", default="waldo_yolov8s_tile256.pt")
    ap.add_argument("--data", default="waldo-data", help="folder with train/ valid/ test/")
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--cell-overlap", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=256, help="YOLO tile size (match the weights)")
    ap.add_argument("--top-k", type=int, default=3, help="how many stripiest cells count as 'high' hints")
    ap.add_argument("--stop-conf", type=float, default=None, help="default: chosen automatically on train")
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--eps-end", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="rl_cache.json")
    ap.add_argument("--q-out", default="q_table_B.npy")
    ap.add_argument("--results-out", default="rl_results_B.json")
    args = ap.parse_args()
    if args.grid != 4:
        print("Note: the state size assumes 4 neighbours, so any grid works, but 4x4 keeps the table small.")
    random.seed(args.seed)
    run_experiment(build_cache(args), args)


if __name__ == "__main__":
    main()
