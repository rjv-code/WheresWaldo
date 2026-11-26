import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from app.detector import get_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "pre-trained-model"
Q_TABLE_PATH = MODEL_DIR / "q_table_B.npy"

GRID_SIZE = 4
CELL_OVERLAP = 0.25
TILE_SIZE = 256
TILE_OVERLAP = 0.25
DETECTION_CONF = 0.05
STOP_CONF = 0.70
TOP_K = 3
MAX_STEPS = 20
RANDOM_SEED = 0


@lru_cache(maxsize=1)
def load_agent_modules():
    """
    Loads Agent B and evaluation modules
    Input/Pre-Condition: Existing pre-trained model scripts
    Output/Post-Condition: Returns the imported modules
    """

    modules = []

    for name in ("rl_tile_search", "evaluate"):
        path = MODEL_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)

        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)

    return tuple(modules)


@lru_cache(maxsize=1)
def load_q_table():
    """
    Loads the trained 30k-episode Q-learning policy
    Input/Pre-Condition: q_table_B.npy exists and has the expected shape
    Output/Post-Condition: Returns Q-table as a NumPy array
    """

    if not Q_TABLE_PATH.is_file():
        raise FileNotFoundError(f"Couldn't find the trained Agent B Q-table: {Q_TABLE_PATH}")

    q_table = np.load(Q_TABLE_PATH, allow_pickle=False)

    if q_table.shape != (1296, 4):
        raise ValueError(f"Q-table shape not expected: {q_table.shape}")

    return q_table


def select_candidate(detections, stripe, width, height, q_table, agent):
    """
    Returns the trained Q-learning policy without ground-truth labels
    Input/Pre-Condition: YOLO detections, stripe scores, image size, trained Q-table,
                         and Agent B module
    Output/Post-Condition: Returns a candidate and grid cells examined
    """

    rects = agent.cell_rects(width, height, GRID_SIZE, CELL_OVERLAP)

    # select the highest-confidence box where the center is in each cell
    best_boxes = [None] * (GRID_SIZE * GRID_SIZE)
    confidence = np.zeros(GRID_SIZE * GRID_SIZE)

    for cell, (x1, y1, x2, y2) in enumerate(rects):
        for detection in detections:
            cx = (detection[0] + detection[2]) / 2
            cy = (detection[1] + detection[3]) / 2

            if x1 <= cx < x2 and y1 <= cy < y2:
                if detection[4] > confidence[cell]:
                    confidence[cell] = detection[4]
                    best_boxes[cell] = detection


    # start at the stripiest cell
    order = np.argsort(-stripe, kind="stable")
    high = np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)

    for cell in order[:TOP_K]:
        high[cell] = stripe[cell] > 0

    checked = np.zeros(GRID_SIZE * GRID_SIZE, dtype=bool)
    position = int(order[0])
    steps = 0
    checks = 0

    # fixed seed to keep web requests reproducible
    rng = np.random.default_rng(RANDOM_SEED)

    while True:
        if not checked[position]:
            checked[position] = True
            checks += 1

            if confidence[position] >= STOP_CONF:
                box = best_boxes[position]

                return {
                    "detection": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                        "confidence": float(box[4])
                    },
                    "regions_examined": checks
                }

            if checked.all():
                break

        if steps >= MAX_STEPS:
            break

        # current cell + four neighboring hints encoded in base 3
        row, col = divmod(position, GRID_SIZE)
        code = 0

        for i, (dr, dc) in enumerate(agent.MOVES):
            nr, nc = row + dr, col + dc

            if (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and not checked[nr * GRID_SIZE + nc]):
                hint = 1 if high[nr * GRID_SIZE + nc] else 0
            else:
                hint = 2

            code += hint * (3 ** i)

        state = position * 81 + code
        action = agent.greedy(q_table[state], rng)

        # move if within bounds
        dr, dc = agent.MOVES[action]
        nr, nc = row + dr, col + dc
        steps += 1

        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
            position = nr * GRID_SIZE + nc

    return {
        "detection": None,
        "regions_examined": checks
    }


def detect_waldo_rl(image):
    """
    Searches an uploaded image using the trained Agent B policy
    Input/Pre-Condition: Validated Flask FileStorage image
    Output/Post-Condition: Returns detections and grid cells examined
    """

    agent, evaluator = load_agent_modules()
    q_table = load_q_table()
    model = get_model(TILE_SIZE)

    image.stream.seek(0)

    with Image.open(image.stream) as source:
        picture = source.convert("RGB")

    width, height = picture.size

    # match YOLO config used to build rl_cache.json
    detections = evaluator.detect(model, picture, TILE_SIZE, TILE_OVERLAP, DETECTION_CONF)

    rects = agent.cell_rects(width, height, GRID_SIZE, CELL_OVERLAP)

    stripe = np.array(agent.stripe_scores(picture, rects), dtype=float)

    result = select_candidate(detections, stripe, width, height, q_table, agent)

    # YOLO scans the whole image before RL policy runs
    stride = int(TILE_SIZE * (1- TILE_OVERLAP))
    tiles_processed = (len(evaluator.tile_starts(width, TILE_SIZE, stride)) * len(evaluator.tile_starts(height, TILE_SIZE, stride)))

    return {
        "detections": ([result["detection"]] if result["detection"] is not None else []),
        "regions_examined": result["regions_examined"],
        "yolo_tiles_processed": tiles_processed,
        "search_method": "q_learning_b"
    }
