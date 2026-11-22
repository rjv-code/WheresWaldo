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
    Output/Post-Condition: Returns a dict with structured search results
"""
def search_image(image):
    # temporary result until integration finished

    detections = [{
        "x1": 100.0,
        "y1": 150.0,
        "x2": 150.0,
        "y2": 230.0,
        "confidence": 0.95
    }]

    return build_search_response(detections, mock = True)