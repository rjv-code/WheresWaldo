from flask import render_template, request, jsonify
from app import create_app

app = create_app()

# limit uploads to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10485760

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_image():
    # check if image was sent
    if "image" not in request.files:
        return jsonify({"error": "No file selected"}), 400

    image = request.files["image"]

    # check file extension
    extension = image.filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Wrong image format"}), 400

    return jsonify({
        "success": True,
        "message": "Image received successfully"
    })

if __name__ == "__main__":
    app.run(debug=True)
