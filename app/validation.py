from flask import current_app

def validate_image(image):
    # check if image provided
    if image is None:
        return "No file selected"

    # check for empty file name
    if not image.filename or not image.filename.strip():
        return "No file selected"

    filename = image.filename.strip()

    # check that filename has an ext
    if "." not in filename:
        return "No file extension"

    # check file ext
    extension = image.filename.rsplit(".", 1)[-1].lower()

    if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
        return "Wrong image format"

    # no validation errors
    return None