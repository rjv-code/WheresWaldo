import io
import pytest

from PIL import Image


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

@pytest.fixture
def mock_search(monkeypatch):
    """Replaces the search controller with predictable test results"""
    def fake_search(image, method="q_learning_b"):
        return {
            "success": True,
            "mock": False,
            "found": True,
            "message": "Waldo may have been found.",
            "detections": [{
                "x1": 100.0,
                "y1": 150.0,
                "x2": 150.0,
                "y2": 230.0,
                "confidence": 0.95
            }],
            "regions_examined": 16
        }

    monkeypatch.setattr("app.routes.search_image", fake_search)


def test_homepage(client):
    """Checks that the homepage loads successfully"""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Finding Waldo" in response.data


def test_upload_valid_image(client, mock_search):
    """Checks that the upload endpoint accepts a valid image"""
    response = client.post("/api/upload", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 200

    result = response.get_json()

    assert result["success"] is True
    assert result["message"] == "Image received successfully"


def test_search_valid_image(client, mock_search):
    """Checks that the search endpoint returns structured results"""
    response = client.post("/api/search", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 200
    
    result = response.get_json()

    assert result["success"] is True
    assert result["mock"] is False
    assert result["found"] is True
    assert isinstance(result["detections"], list)


def test_search_detection_structure(client, mock_search):
    """Checks the bounding-box format returned by controller"""
    response = client.post("/api/search", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 200
    
    result = response.get_json()
    detections = result["detections"]

    assert len(detections) > 0

    detection = detections[0]

    assert "x1" in detection
    assert "y1" in detection
    assert "x2" in detection
    assert "y2" in detection
    assert "confidence" in detection

    assert detection["x1"] < detection["x2"]
    assert detection["y1"] < detection["y2"]
    assert 0 <= detection["confidence"] <= 1


def test_search_rejects_invalid_method(client):
    """
    Checks API validation of the selected search method
    Input/Pre-Condition: Valid image and unsupported method
    Output/Post-Condition: Error 400 w/structured response
    """
    response = client.post(
        "/api/search",
        data={
            "image": (create_test_image(), "waldo.png"),
            "method": "invalid_method"
        },
        content_type="multipart/form-data"
    )

    assert response.status_code == 400

    result = response.get_json()

    assert result["success"] is False
    assert result["error"] == "INVALID_SEARCH_METHOD"
