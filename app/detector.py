import importlib.util
import tempfile
from functools import lru_cache
from pathlib import Path

from flask import current_app


# locate existing detector script
PROJECT_ROOT= Path(__file__).resolve().parent.parent
DETECTOR_PATH = PROJECT_ROOT / "pre-trained-model" / "find_waldo.py"


@lru_cache(maxsize=1)
def load_detector_module():
    """
    Loads the pretrained detector module
    Input/Pre-Condition: Existing find_waldo.py script
    Output/Post-Condition: Returns the imported detector module
    """
    spec = importlib.util.spec_from_file_location("waldo_detector", DETECTOR_PATH)

    if spec is None or spec.loader is None:
        raise ImportError("Unable to load YOLOv8 detector module")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@lru_cache(maxsize=5)
def get_model(tile_size=512):
    """
    Loads and caches a pretrained YOLOv8 model
    Input/Pre-Condition: Supported tile size
    Output/Post-Condition: Returns a loaded YOLOv8 model
    """
    detector = load_detector_module()
    return detector.load_model(tile_size)


def detect_waldo(image, tile_size=512, overlap=0.125, conf=0.1):
    """
    Runs YOLOv8 detection on an uploaded image
    Input/Pre-Condition: Validated Flask FileStorage image
    Output/Post-Condition: Returns prediced Waldo bounding boxes
    """
    detector = load_detector_module()
    model = get_model(tile_size)

    # temporarily save uploaded file since existing detector expects a path
    image.stream.seek(0)

    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "upload.png"

        image.save(image_path)

        detections = detector.find_waldo(
            str(image_path),
            model,
            tile_size=tile_size,
            overlap=overlap,
            conf=conf
        )

    return detections