from flask import current_app

def validate_image(image):
    # check if image provided
    if image is None:
        return "No file selected"

    # check file ext
    extension = image.filename.rsplit(".", 1)[-1].lower()

    if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
        return "Wrong image format"

    # no validation errors
    return None