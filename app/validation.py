import warnings

from flask import current_app
from PIL import Image, UnidentifiedImageError

def validate_image(image):
    # check if image provided
    if image is None:
        return ("MISSING_IMAGE", "No file selected")

    # check for empty file name
    if not image.filename or not image.filename.strip():
        return ("MISSING_IMAGE", "No file selected")

    filename = image.filename.strip()

    # check that filename has an ext
    if "." not in filename:
        return ("MISSING_EXTENSION", "No file extension")

    # check file ext
    extension = image.filename.rsplit(".", 1)[-1].lower()

    if extension not in current_app.config["ALLOWED_EXTENSIONS"]:
        return ("UNSUPPORTED_FORMAT", "Wrong image format")

    # check file size
    image.stream.seek(0, 2)
    file_size = image.stream.tell()
    image.stream.seek(0)

    # reject empty files
    if file_size == 0:
        return ("EMPTY_IMAGE", "File is empty")

    # reject too large images
    if file_size > current_app.config["MAX_IMAGE_SIZE"]:
        return ("IMAGE_TOO_LARGE", "Image is larger than 10 MB")

    # match file extensions w/actual image formats
    expected_formats = {
        "png": "PNG",
        "jpg": "JPEG",
        "jpeg": "JPEG",
        "webp": "WEBP"
    }

    try:
        # decompression bombs are validation errors
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)

            # inspect uploaded image
            with Image.open(image.stream) as img:
                actual_format = img.format

                # confirm actual format matches extension
                if actual_format != expected_formats[extension]:
                    return ("FORMAT_MISMATCH", "Image format doesn't match file extension")

                # verify image integrity
                img.verify()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        return ("INVALID_IMAGE", "Invalid or corrupted image")
    finally:
        # reset stream for future image processing
        image.stream.seek(0)


    # no validation errors
    return None