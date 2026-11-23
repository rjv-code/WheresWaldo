from time import perf_counter

from evaluation.metrics import evaluate_detection

def evaluate_image(case, search_function):
    """
    Evaluates a single image search attempt
    Input/Pre-Condition: A labeled test case and a search function that accepts the test case and returns results
    Output/Post-Condition: Returns detection results and search metrics
    """
    start_time = perf_counter()
    result = search_function(case)
    search_time = perf_counter() - start_time

    # prevent simulated detections from being reported as model results
    if result.get("mock", False):
        raise ValueError("Can't evaluate simulated search results")

    detections = result.get("detections", [])

    # use highest-confidence detection as final prediction
    predicted = (max(detections, key=lambda detection: detection["confidence"]) if detections else None)

    actual = case["actual"]
    evaluation = evaluate_detection(predicted, actual)

    return {
        "image_id": case["image_id"],
        "found": predicted is not None,
        "iou": evaluation["iou"],
        "classification": evaluation["classification"],
        "reward": evaluation["reward"],
        "regions_examined": result.get("regions_examined"),
        "search_time_seconds": search_time
    }


def evaluate_images(cases, search_function):
    """
    Evaluates multiple labeled image search attempts
    Input/Pre-Condition: List of test cases and callable search function
    Output/Post-Condition: Returns a list of evaluation results
    """
    results = []

    for case in cases:
        result = evaluate_image(case, search_function)
        results.append(result)

    return results