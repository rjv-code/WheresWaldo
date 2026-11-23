def calculate_iou(predicted, actual):
    """
    Calculates Intersection over Union between two bounding boxes.
    Input/Pre-Condition: Two dictionaries with x1, y1, x2, and y2
    Output/Post-Condition: Returns an IoU score between 0 and 1
    """
    x_left = max(predicted["x1"], actual["x1"])
    y_top = max(predicted["y1"], actual["y1"])
    x_right = min(predicted["x2"], actual["x2"])
    y_bottom = min(predicted["y2"], actual["y2"])

    intersection_width = max(0, x_right - x_left)
    intersection_height = max(0, y_bottom - y_top)
    intersection_area = intersection_width * intersection_height

    predicted_width = max(0, predicted["x2"] - predicted["x1"])
    predicted_height = max(0, predicted["y2"] - predicted["y1"])
    predicted_area = predicted_width * predicted_height

    actual_width = max(0, actual["x2"] - actual["x1"])
    actual_height = max(0, actual["y2"] - actual["y1"])
    actual_area = actual_width * actual_height

    union_area = predicted_area + actual_area - intersection_area

    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def evaluate_detection(predicted, actual):
    """
    Evaluates a Waldo prediction using IoU and our reward rules.
    Input/Pre-Condition: Predicted & actual bounding boxes, or None
                         if Waldo wasn't detected
    Output/Post-Condition: Returns the IoU, classification, and reward
    """
    if actual is None and predicted is None:
        return {"iou": None, "classification": "correct", "reward": 1.0}

    if actual is None or predicted is None:
        return {"iou": 0.0, "classification": "incorrect", "reward": -1.0}

    iou = calculate_iou(predicted, actual)

    if iou >= 0.5:
        classification = "correct"
        reward = 1.0
    elif iou >= 0.25:
        classification = "partial"
        reward = 0.5
    else:
        classification = "incorrect"
        reward = -1.0

    return {
        "iou": iou,
        "classification": classification,
        "reward": reward
    }