import io

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
        "x2": 100.0,
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


def test_search_image_placeholder():
    """Checks that the placeholder controller returns structured results"""
    result = search_image(create_test_image())

    assert result["success"] is True
    assert result["mock"] is True
    assert result["found"] is True

    assert len(result["detections"]) == 1

    detection = result["detections"][0]
    assert detection["confidence"] == 0.95


def test_search_controller_failure(client, monkeypatch):
    """
    Checks that unexpected controller failures return Error 500
    Input/Pre-Condition: Valid image and simulated controller exception
    Output/Post-Condition: Returns a structured SEARCH_FAILED error
    """
    def failing_search(image):
        raise RuntimeError("Simulated search failure")

    monkeypatch.setattr("app.routes.search_image", failing_search)

    response = client.post("/api/search", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 500

    result = response.get_json()

    assert result["success"] is False
    assert result["error"] == "SEARCH_FAILED"
    assert isinstance(result["message"], str)
    