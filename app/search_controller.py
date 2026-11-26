from app.rl_detector import detect_waldo_rl

def build_search_response(detections, mock = True, regions_examined = None):
    """ 
    Creates a consistent response for image searches
    Input/Pre-Condition: Takes a list of detection dictionaries and a bool for if the detections are simulated or not
    Output/Post-Condition: Returns a dict with structured search results
    """
    found = len(detections) > 0

    if mock:
        message = ("Simulated detection, no AI detection" if found else "Simulated non-detection, no AI detection")
    else:
        message = ("Waldo may have been found." if found else "No Waldo detected.")

    return {
        "success": True,
        "mock": mock,
        "found": found,
        "message": message,
        "detections": detections,
        "regions_examined": regions_examined
    }


def search_image(image):
    """ 
    Coordinates an image search request using Q-learning Agent B
    Input/Pre-Condition: Accepts an image, which is a validated Flask FileStorage object
    Output/Post-Condition: Returns detections, grid cells examined, and YOLO inference info
    """
    result = detect_waldo_rl(image)

    response = build_search_response(result["detections"], mock=False, regions_examined=result["regions_examined"])
    response["search_method"] = result["search_method"]
    response["yolo_tiles_processed"] = result["yolo_tiles_processed"]

    return response
