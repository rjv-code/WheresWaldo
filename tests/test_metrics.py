import pytest

from evaluation.metrics import calculate_iou, evaluate_detection

BOX = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}


@pytest.mark.parametrize(
    "predicted, actual, expected",
    [
        (BOX, BOX, 1.0),
        ({"x1": 100, "y1": 100, "x2": 200, "y2": 200}, BOX, 0.0),
        ({"x1": 0, "y1": 0, "x2": 50, "y2": 100}, BOX, 0.5),
        ({"x1": 0, "y1": 0, "x2": 25, "y2": 100}, BOX, 0.25),
    ]
)
def test_calculate_iou(predicted, actual, expected):
    """Checks IoU calculations against known bounding boxes"""
    assert calculate_iou(predicted, actual) == pytest.approx(expected)


def test_iou_zero_area():
    """Checks that no-area boxes don't cause division errors"""
    predicted = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

    assert calculate_iou(predicted, BOX) == 0.0


@pytest.mark.parametrize(
    "predicted, actual, expected_classification, expected_reward",
    [
        (BOX, BOX, "correct", 1.0),
        ({"x1": 0, "y1": 0, "x2": 50, "y2": 100}, BOX, "correct", 1.0),
        ({"x1": 0, "y1": 0, "x2": 25, "y2": 100}, BOX, "partial", 0.5),
        ({"x1": 200, "y1": 200, "x2": 300, "y2": 300}, BOX, "incorrect", -1.0),
        (None, BOX, "incorrect", -1.0),
        (BOX, None, "incorrect", -1.0),
        (None, None, "correct", 1.0),
    ]
)
def test_evaluate_detection(predicted, actual, expected_classification, expected_reward):
    """Checks classification and reward rules for Waldo predictions"""
    result = evaluate_detection(predicted, actual)

    assert result["classification"] == expected_classification
    assert result["reward"] == expected_reward


def test_absent_waldo_has_no_iou():
    """Checks that a correct absence prediction has no IoU value"""
    result = evaluate_detection(None, None)

    assert result["iou"] is None