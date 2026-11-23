def test_app_config(app):
    assert app.config["TESTING"] is True
    assert app.config["MAX_IMAGE_SIZE"] == 10 * 1024 * 1024

def test_client_available(client):
    response = client.get("/")
    assert response.status_code == 200