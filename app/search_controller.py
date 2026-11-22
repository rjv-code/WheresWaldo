""" Coordinates an image search request.
    Input/Pre-Condition: Accepts an image, which is a validated Flask FileStorage object
    Output/Post-Condition: Returns a dict with structured search results
"""
def search_image(image):
    # temporary result until integration finished

    return {
        "success": True,
        "mock": True,
        "found": True,
        "message": "Simulated detection, no AI detection",
        "detections": [{
            "x1": 100.0,
            "y1": 150.0,
            "x2": 150.0,
            "y2": 230.0,
            "confidence": 0.95
        }]
    }