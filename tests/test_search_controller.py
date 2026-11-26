import io
import pytest

from PIL import Image

from app.search_controller import build_search_response, search_image


def create_test_image():
    """
    Creates a valid PNG image for testing
    Input/Pre-Condition: void
    Output/Post-Condition: Returns in-memory image stream
    """
    image = Image.new("RGB", (100, 100), color="white")

    image_stream = io.BytesIO()
    image.save(image_stream, format="PNG")
    image_stream.seek(0)

    return image_stream


def test_search_response_with_detections():
    """Checks that detections produce a successful search response."""
    detections = [{
        "x1": 100.0,
        "y1": 150.0,
        "x2": 150.0,
        "y2": 230.0,
        "confidence": 0.95
    }]

    result = build_search_response(detections, mock=True)

    assert result["success"] is True
    assert result["mock"] is True
    assert result["found"] is True
    assert result["detections"] == detections
    assert isinstance(result["message"], str)


def test_search_response_without_detections():
    """Checks that an empty detection list says that Waldo wasn't found"""
    result = build_search_response([], mock=True)

    assert result["success"] is True
    assert result["mock"] is True
    assert result["found"] is False
    assert result["detections"] == []


def test_search_response_without_mock():
    """Checks that real search results get correctly identified"""
    result = build_search_response([], mock=False)
    
    assert result["success"] is True
    assert result["mock"] is False
    assert result["found"] is False
    assert result["message"] == "No Waldo detected."


def test_search_image_uses_detector(monkeypatch):
    """
    Checks that the search controller uses YOLOv8 detections
    Input/Pre-Condition: Valid image and simulated detector output
    Output/Post-Condition: Returns structured non-mock search results
    """
    detections = [{
        "x1": 100.0,
        "y1": 150.0,
        "x2": 150.0,
        "y2": 230.0,
        "confidence": 0.95
    }]

    def fake_detector(image):
        return {
            "detections": detections,
            "regions_examined": 5,
            "yolo_tiles_processed": 100,
            "search_method": "q_learning_b"
        }

    monkeypatch.setattr("app.search_controller.detect_waldo_rl", fake_detector)

    result = search_image(create_test_image())

    assert result["success"] is True
    assert result["mock"] is False
    assert result["found"] is True
    assert result["detections"] == detections

    assert len(result["detections"]) == 1
    assert result["detections"][0]["confidence"] == 0.95

    # agent B test
    assert result["regions_examined"] == 5
    assert result["yolo_tiles_processed"] == 100
    assert result["search_method"] == "q_learning_b"


def test_search_controller_failure(client, monkeypatch):
    """
    Checks that unexpected controller failures return Error 500
    Input/Pre-Condition: Valid image and simulated controller exception
    Output/Post-Condition: Returns a structured SEARCH_FAILED error
    """
    def failing_search(image, method="q_learning_b"):
        raise RuntimeError("Simulated search failure")

    monkeypatch.setattr("app.routes.search_image", failing_search)

    response = client.post("/api/search", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 500

    result = response.get_json()

    assert result["success"] is False
    assert result["error"] == "SEARCH_FAILED"
    assert isinstance(result["message"], str)


def test_search_image_without_detections(monkeypatch):
    """
    Checks how the controller handles an unsuccessful Agent B search
    Input/Pre-Condition: Simulated Agent B output without detections
    Output/Pre-ConditionL Returns a non-mock response showing no detection
    """

    def fake_detector(image):
        return {
            "detections": [],
            "regions_examined": 2,
            "yolo_tiles_processed": 100,
            "search_method": "q_learning_b"
        }

    monkeypatch.setattr("app.search_controller.detect_waldo_rl", fake_detector)

    result = search_image(create_test_image())

    assert result["success"] is True
    assert result["mock"] is False
    assert result["found"] is False
    assert result["detections"] == []
    assert result["message"] == "No Waldo detected."

    assert result["regions_examined"] == 2
    assert result["yolo_tiles_processed"] == 100
    assert result["search_method"] == "q_learning_b"


def test_search_image_uses_yolo_baseline(monkeypatch):
    """
    Checks that selecting YOLO uses the original detector
    Input/Pre-Condition: Chosen YOLO method and simulated results
    Output/Post-Condition: Returns YOLO results w/method metadata
    """
    detections = [{
        "x1": 25.0,
        "y1": 30.0,
        "x2": 70.0,
        "y2": 90.0,
        "confidence": 0.88
    }]

    def fake_yolo(image):
        return {
            "detections": detections,
            "regions_examined": 25
        }

    def unexpected_rl(image):
        raise AssertionError("Agent B shouldn't be called")

    monkeypatch.setattr("app.search_controller.detect_waldo", fake_yolo)
    monkeypatch.setattr("app.search_controller.detect_waldo_rl", unexpected_rl)

    result = search_image(create_test_image(), method="yolo")

    assert result["success"] is True
    assert result["found"] is True
    assert result["mock"] is False
    assert result["detections"] == detections
    assert result["search_method"] == "yolo"
    assert result["regions_examined"] == 25


def test_search_image_rejects_invalid_method():
    """
    Checks that unsupported search methods are rejected
    Input/Pre-Condition: An unsupported method name
    Output/Post-Condition: ValueError
    """
    with pytest.raises(ValueError, match="Unsupported search method"):
        search_image(create_test_image(), method="invalid_method")