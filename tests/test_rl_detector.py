from types import SimpleNamespace

import numpy as np
import pytest

from app.rl_detector import select_candidate


@pytest.fixture
def fake_agent():
    """Creates a simplified Agent B interface for testing"""
    def cell_rects(width, height, grid, overlap):
        cell_width = width / grid
        cell_height = height / grid
        rects = []

        for row in range(grid):
            for col in range(grid):
                x = col * cell_width
                y = row * cell_height


                rects.append((
                    max(0, x - overlap * cell_width),
                    max(0, y - overlap * cell_height),
                    min(width, x + cell_width * (1 + overlap)),
                    min(height, y + cell_height * (1 + overlap))
                ))

        return rects

    return SimpleNamespace(cell_rects=cell_rects, MOVES=[(-1, 0), (1, 0), (0, -1), (0, 1)], greedy=lambda row, rng: int(np.argmax(row)))


@pytest.fixture
def q_table():
    """Creates a controlled Q-table"""
    return np.zeros((1296, 4))


@pytest.fixture
def stripe_scores():
    """Makes the top-left grid cell the start position"""
    return np.array([1.0] + [0.0] * 15)


def test_agent_finds_candidate_in_starting_cell(fake_agent, q_table, stripe_scores):
    """
    Checks condition in start cell
    Input/Pre-Condition: Candidate confidence above 70%
    Output/Post-Condition: Returns candidate after one cell check
    """

    detection = [40, 40, 60, 60, 0.90]

    result = select_candidate([detection], stripe_scores, 400, 400, q_table, fake_agent)

    assert result["detection"] is not None
    assert result["detection"]["confidence"] == 0.90
    assert result["regions_examined"] == 1


def test_agent_moves_to_candidate(fake_agent, q_table, stripe_scores):
    """
    Checks that Q-values guide movement between grid cells
    Input/Pre-Condition: Agent starts in cell 0 and moves right
    Output/Post-Condition: Candidate should be found in cell 1
    """

    # start state: cell 0 w/neighboring hints
    # state index 20 is the first checked cell
    q_table[20, 3] = 1.0

    detection = [150, 40, 170, 60, 0.85]

    result = select_candidate([detection], stripe_scores, 400, 400, q_table, fake_agent)

    assert result["detection"] is not None
    assert result["detection"]["confidence"] == 0.85
    assert result["regions_examined"] == 2


def test_agent_rejects_low_confidence(fake_agent, q_table, stripe_scores):
    """
    Checks that confidence below the threshold isn't enough
    Input/Pre-Condition: Confidence below 70%
    Output/Post-Condition: No detection
    """
    q_table[20, 0] = 1.0

    detection = [40, 40, 60, 60, 0.69]

    result = select_candidate([detection], stripe_scores, 400, 400, q_table, fake_agent)

    assert result["detection"] is None
    assert result["regions_examined"] == 1


def test_agent_stops_after_maximum_steps(fake_agent, q_table, stripe_scores):
    """
    Checks that the search ends after the movement limit
    Input/Pre-Condition: No detections and repeated invalid moves
    Output/Post-Condition: Agent ends without candidate
    """
    q_table[20, 0] = 1.0

    moves = 0

    def count_moves(row, rng):
        nonlocal moves
        moves += 1
        return int(np.argmax(row))

    fake_agent.greedy = count_moves

    result = select_candidate([], stripe_scores, 400, 400, q_table, fake_agent)

    assert result["detection"] is None
    assert result["regions_examined"] == 1
    assert moves == 20
