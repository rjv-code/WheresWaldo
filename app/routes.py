from flask import Blueprint, render_template, request, jsonify
from app.validation import validate_image

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/api/upload", methods = ["POST"])
def upload_image():
    image = request.files.get("image")

    # perform validation on image
    error = validate_image(image)

    if error:
        return jsonify({"error": error}), 400

    return jsonify({"success": True, "message": "Image received successfully"})

@main.route("/api/search", methods = ["POST"])
def search():
    image = request.files.get("image")

    # perform validation on image
    error = validate_image(image)
    
    if error:
        return jsonify({"error": error}), 400

    # temp until search controller implemented
    return jsonify({
        "success": True,
        "mock": True,
        "message": "Search endpoint is operating fine",
        "detections": []
    }), 200