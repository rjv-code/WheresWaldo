import io

import pytest
from PIL import Image

ENDPOINTS = ["/api/upload", "/api/search"]


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


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_missing_image(client, endpoint):
    """Checks that requests without an image are rejected"""
    response = client.post(endpoint)

    assert response.status_code == 400
    assert response.get_json()["error"] == "MISSING_IMAGE"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_empty_filename(client, endpoint):
    """Checks that empty image filenames are rejected"""
    response = client.post(endpoint, data = {"image": (create_test_image(), "")}, content_type = "multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "MISSING_IMAGE"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_missing_extension(client, endpoint):
    """Checks that filenames without extensions are rejected"""
    response = client.post(endpoint, data = {"image": (create_test_image(), "waldo")}, content_type = "multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "MISSING_EXTENSION"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_unsupported_extension(client, endpoint):
    """Checks that unsupported file extensions are rejected"""
    response = client.post(endpoint, data = {"image": (create_test_image(), "waldo.pdf")}, content_type = "multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "UNSUPPORTED_FORMAT"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_empty_image(client, endpoint):
    """Checks that empty image files are rejected"""
    response = client.post(endpoint, data = {"image": (io.BytesIO(b""), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "EMPTY_IMAGE"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_corrupted_image(client, endpoint):
    """Checks that corrupted image files are rejected"""
    response = client.post(endpoint, data = {"image": (io.BytesIO(b"corrupted image"), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "INVALID_IMAGE"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_mismatched_image_format(client, endpoint):
    """Checks that image files that don't match their file extensions are rejected"""
    response = client.post(endpoint, data = {"image": (create_test_image(), "waldo.jpg")}, content_type = "multipart/form-data")

    assert response.status_code == 400
    assert response.get_json()["error"] == "FORMAT_MISMATCH"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_oversized_image(client, endpoint, app):
    """Checks that oversized uploads are rejected and return Error 413"""
    max_request_size = app.config["MAX_CONTENT_LENGTH"]

    response = client.post(endpoint, data = {"image": (io.BytesIO(b"x" * (max_request_size + 1)), "waldo.png")}, content_type = "multipart/form-data")

    assert response.status_code == 413

    result = response.get_json()
    assert result["success"] is False
    assert result["error"] == "IMAGE_TOO_LARGE"