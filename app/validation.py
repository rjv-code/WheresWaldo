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

    # check file size
    image.stream.seek(0, 2)
    file_size = image.stream.tell()
    image.stream.seek(0)

    # reject empty files
    if file_size == 0:
        return "File is empty"

    # reject too large images
    if file_size > current_app.config["MAX_IMAGE_SIZE"]:
        return "Image is larger than 10 MB"

    # no validation errors
    return None