"""
The environment wraps a Where's Waldo image and the pretrained YOLO model.
Each episode is one image. The agent picks tiles to inspect one at a time;
the environment runs YOLO on that tile and returns (next_state, reward, done).

Tile states:
    0 = unchecked
    1 = checked, no Waldo (confidence < found_threshold)
    2 = checked, high confidence (Waldo found → episode over)

Usage:
    env = WaldoEnv(model, found_threshold=0.3)
    state = env.reset("puzzle.jpg")
    next_state, reward, done, info = env.step(action)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image
from ultralytics import YOLO


# Tile status constants — readable aliases
UNCHECKED = 0
NO_WALDO  = 1
FOUND     = 2

# Rewards
REWARD_FOUND     =  10.0   # Waldo detected in this tile
REWARD_NO_WALDO  =  -1.0   # Tile checked, nothing found
REWARD_EXHAUSTED =  -5.0   # All tiles checked with no Waldo


class WaldoEnv:


    def __init__(
        self,
        model: YOLO,
        tile_size: int = 512,
        overlap: float = 0.125,
        found_threshold: float = 0.3,
        conf: float = 0.1,
    ) -> None:
        self.model           = model
        self.tile_size       = tile_size
        self.overlap         = overlap
        self.found_threshold = found_threshold
        self.conf            = conf

        # Set by reset()
        self._image:      Optional[Image.Image] = None
        self._tile_coords: list[tuple[int, int]] = []   # (x, y) top-left of each tile
        self._tile_status: list[int]              = []   # UNCHECKED / NO_WALDO / FOUND
        self._n_tiles:    int                     = 0

    # Public API

    def reset(self, image_path: str) -> tuple[int, ...]:
        """
        Start a new episode on the given image.

        Returns
        -------
        state : tuple of ints (one per tile, all UNCHECKED at start)
        """
        self._image = Image.open(image_path).convert("RGB")
        self._tile_coords = self._compute_tiles()
        self._n_tiles     = len(self._tile_coords)
        self._tile_status = [UNCHECKED] * self._n_tiles

        img_w, img_h = self._image.size
        print(
            f"  Env reset: {Path(image_path).name}  "
            f"({img_w}×{img_h}px)  →  {self._n_tiles} tiles"
        )
        return self._state()

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        """
        Inspect the tile at index `action`.

        Parameters
        ----------
        action : int — tile index (must be in valid_actions())

        Returns
        -------
        next_state : tuple[int, ...]
        reward     : float
        done       : bool
        info       : dict with 'confidence', 'tile_xy', 'found'
        """
        assert 0 <= action < self._n_tiles, f"Action {action} out of range"
        assert self._tile_status[action] == UNCHECKED, (
            f"Tile {action} already checked — only pick from valid_actions()"
        )

        x, y = self._tile_coords[action]
        confidence = self._run_yolo(x, y)

        if confidence >= self.found_threshold:
            self._tile_status[action] = FOUND
            reward = REWARD_FOUND
            done   = True
        else:
            self._tile_status[action] = NO_WALDO
            # All tiles exhausted?
            if all(s != UNCHECKED for s in self._tile_status):
                reward = REWARD_EXHAUSTED
                done   = True
            else:
                reward = REWARD_NO_WALDO
                done   = False

        info = {
            "confidence": confidence,
            "tile_xy":    (x, y),
            "found":      confidence >= self.found_threshold,
            "tiles_checked": sum(s != UNCHECKED for s in self._tile_status),
            "n_tiles":    self._n_tiles,
        }
        return self._state(), reward, done, info

    def valid_actions(self) -> list[int]:
        """Return indices of tiles that have not been checked yet."""
        return [i for i, s in enumerate(self._tile_status) if s == UNCHECKED]

    def n_tiles(self) -> int:
        return self._n_tiles

    # Private helpers

    def _state(self) -> tuple[int, ...]:
        """Current state as an immutable tuple (Q-table key)."""
        return tuple(self._tile_status)

    def _compute_tiles(self) -> list[tuple[int, int]]:
        """Return (x, y) top-left corners for all tiles covering the image."""
        img_w, img_h = self._image.size
        ts     = self.tile_size
        stride = int(ts * (1 - self.overlap))
        coords = []
        for y in range(0, img_h - ts + 1, stride):
            for x in range(0, img_w - ts + 1, stride):
                coords.append((x, y))
        return coords

    def _run_yolo(self, x: int, y: int) -> float:
        """
        Crop the tile at (x, y) and run YOLO.
        Returns the highest confidence detection found, or 0.0 if none.
        """
        ts   = self.tile_size
        tile = self._image.crop((x, y, x + ts, y + ts))
        results = self.model.predict(
            tile, conf=self.conf, verbose=False, imgsz=ts
        )
        best_conf = 0.0
        for result in results:
            for box in result.boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf = c
        return best_conf
