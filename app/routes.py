from flask import Blueprint, render_template, request, jsonify
from app.validation import validate_image

main = Blueprint("main", __name__)


def validation_error_response(error):
    # convert validation error into JSON response
    error_code, message = error

    status_code = 413 if error_code == "IMAGE_TOO_LARGE" else 400

    return jsonify({"success": False, "error": error_code, "message": message}), status_code


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/api/upload", methods = ["POST"])
def upload_image():
    image = request.files.get("image")

    # perform validation on image
    error = validate_image(image)

    if error:
        return validation_error_response(error)

    return jsonify({"success": True, "message": "Image received successfully"})


@main.route("/api/search", methods = ["POST"])
def search():
    image = request.files.get("image")

    # perform validation on image
    error = validate_image(image)
    
    if error:
        return validation_error_response(error)

    # temp until search controller implemented
    return jsonify({
        "success": True,
        "mock": True,
        "message": "Search endpoint is operating fine",
        "detections": []
    }), 200