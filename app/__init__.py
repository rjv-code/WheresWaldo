from flask import Flask

def create_app():
    app = Flask(__name__)

    # app config
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}

    # register app routes
    from app.routes import main
    app.register_blueprint(main)

    return app