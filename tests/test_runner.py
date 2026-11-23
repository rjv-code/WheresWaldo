import pytest

from evaluation.runner import evaluate_image, evaluate_images

BOX = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}


def test_evaluate_correct_detection():
    """Checks evaluation of a correct Waldo prediction"""
    case = {"image_id": "image1.png", "actual": BOX}

    def search_function(case):
        return {
            "found": True,
            "mock": False,
            "detections": [{**BOX, "confidence": 0.95}],
            "regions_examined": 5
        }

    result = evaluate_image(case, search_function)

    assert result["image_id"] == "image1.png"
    assert result["found"] is True
    assert result["waldo_present"] is True
    assert result["iou"] == pytest.approx(1.0)
    assert result["classification"] == "correct"
    assert result["reward"] == 1.0
    assert result["regions_examined"] == 5
    assert result["search_time_seconds"] >= 0


def test_evaluate_missing_detection():
    """Checks evaluation when Waldo is present but not detected."""
    case = {"image_id": "image2.png", "actual": BOX}
    
    def search_function(case):
        return {
            "found": False,
            "mock": False,
            "detections": [],
            "regions_examined": 10
        }

    result = evaluate_image(case, search_function)

    assert result["found"] is False
    assert result["iou"] == 0.0
    assert result["classification"] == "incorrect"
    assert result["reward"] == -1.0


def test_evaluate_absent_waldo():
    """Checks evaluation when Waldo is correctly predicted absent"""
    case = {"image_id": "image2.png", "actual": None}
        
    def search_function(case):
        return {
            "found": False,
            "mock": False,
            "detections": [],
            "regions_examined": 8
        }

    result = evaluate_image(case, search_function)

    assert result["found"] is False
    assert result["waldo_present"] is False
    assert result["iou"] is None
    assert result["classification"] == "correct"
    assert result["reward"] == 1.0


def test_evaluate_multiple_images():
    """Checks that multiple test cases are evaluated"""
    cases = [
        {"image_id": "image1.png", "actual": BOX},
        {"image_id": "image2.png", "actual": None}
    ]

    def search_function(case):
        return {
            "mock": False,
            "detections": (
                [{**BOX, "confidence": 0.95}] if case["image_id"] == "image1.png" else []
            ),
            "regions_examined": 4
        }

    results = evaluate_images(cases, search_function)

    assert len(results) == 2
    assert results[0]["classification"] == "correct"
    assert results[1]["classification"] == "correct"


def test_reject_simulated_results():
    """Checks that mock controller results aren't evaluated"""
    case = {"image_id": "image1.png", "actual": BOX}

    def search_function(case):
        return {
            "mock": True,
            "detections": [{**BOX, "confidence": 0.95}]
        }

    with pytest.raises(ValueError, match = "simulated"):
        evaluate_image(case, search_function)