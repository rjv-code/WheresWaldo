from flask import Blueprint, render_template, request, jsonify, current_app

main = Blueprint("main", __name__)

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/api/upload", methods = ["POST"])
def upload_image():
    # check if an image was sent
    if "image" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    image = request.files["image"]

    # check file extension
    extension = image.filename.rsplit(".", 1)[-1].lower()

    if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
        return jsonify({"error": "Wrong image format"}), 400

    return jsonify({"success": True, "message": "Image received successfully"})

@main.route("/api/search", methods = ["POST"])
def search():
    # temp until search controller implemented
    return jsonify({
        "success": True,
        "mock": True,
        "message": "Search endpoint is operating fine",
        "detections": []
    }), 200