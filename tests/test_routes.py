import io

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


def test_homepage(client):
    """Checks that the homepage loads successfully"""
    response = client.get("/")

    assert response.status_code == 200
    assert b"Finding Waldo" in response.data


def test_upload_valid_image(client):
    """Checks that the upload endpoint accepts a valid image"""
    response = client.post("/api/upload", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 200

    result = response.get_json()

    assert result["success"] is True
    assert result["message"] == "Image received successfully"


def test_search_valid_image(client):
    """Checks that the search endpoint returns structured results"""
    response = client.post("/api/search", data = {"image": (create_test_image(), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 200
    
    result = response.get_json()

    assert result["success"] is True
    assert result["mock"] is True
    assert result["found"] is True
    assert isinstance(result["detections"], list)


def test_search_detection_structure(client):
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
