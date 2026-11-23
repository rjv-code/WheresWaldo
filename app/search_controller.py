from app.detector import detect_waldo

""" Creates a consistent response for image searches
    Input/Pre-Condition: Takes a list of detection dictionaries and a bool for if the detections are simulated or not
    Output/Post-Condition: Returns a dict with structured search results
"""
def build_search_response(detections, mock = True):
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
        "detections": detections
    }


""" Coordinates an image search request.
    Input/Pre-Condition: Accepts an image, which is a validated Flask FileStorage object
    Output/Post-Condition: Returns a dict with structured search results from YOLOv8
"""
def search_image(image):
    detections = detect_waldo(image)

    return build_search_response(detections, mock = False)