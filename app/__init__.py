from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

def create_app():
    app = Flask(__name__)

    # app config
    app.config["MAX_IMAGE_SIZE"] = 10 * 1024 * 1024                                     # 10 MB max image
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_IMAGE_SIZE"] + 64 * 1024         # additional space for HTTP metadata
    app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}

    # return JSON when request exceeds upload limits
    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(error):
        return jsonify({"success": False, "error": "Upload is larger than 10 MB"}), 413

    # register app routes
    from app.routes import main
    app.register_blueprint(main)

    return app