from flask import Blueprint, render_template, request, jsonify

main = Blueprint("main", __name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

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

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Wrong image format"}), 400

    return jsonify({"success": True, "message": "Image received successfully"})
