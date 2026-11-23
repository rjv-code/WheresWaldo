def summarize_results(results):
    """
    Summarizes the results of multiple image searches.
    Input/Pre-Condition: List of evaluation result dictionaries
    Output/Post-Condition: Returns a dictionary of overall metrics
    """
    total = len(results)

    if total == 0:
        return {
            "total_images": 0,
            "correct_detections": 0,
            "detection_success_rate": 0.0,
            "average_iou": None,
            "average_regions_examined": None,
            "average_search_time": None,
            "average_reward": None
        }

    correct = sum(result["classification"] == "correct" for result in results)

    # only include appropriate IoU metrics
    iou_values = [result["iou"] for result in results if result["waldo_present"] and result["iou"] is not None]

    # exclude missing region counts instead of treating as 0
    region_values = [result["regions_examined"] for result in results if result.get("regions_examined") is not None]

    time_values = [result["search_time_seconds"] for result in results if result.get("search_time_seconds") is not None]

    total_reward = sum(result["reward"] for result in results)

    return {
        "total_images": total,
        "correct_detections": correct,
        "detection_success_rate": correct / total * 100,
        "average_iou": (sum(iou_values) / len(iou_values) if iou_values else None),
        "average_regions_examined": (sum(region_values) / len(region_values) if region_values else None),
        "average_search_time": (sum(time_values) / len(time_values) if time_values else None),
        "average_reward": total_reward / total
    }


def print_report(summary):
    """
    Prints a formatted evaluation summary to the console
    Input/Pre-Condition: Dictionary returned by summarize_results()
    Output/Post-Condition: Shows the eval metrics
    """

    def format_average(value, suffix=""):
        return f"{value:.2f}{suffix}" if value is not None else "N/A"

    title = "Finding Waldo - Evaluation Results"
    width = 46

    print(f"\n{'-' * width}")
    print(f"{title:^{width}}")
    print(f"{'-' * width}")

    print(f"{'Images evaluated:':<28}{summary['total_images']:>18}")
    print(f"{'Correct detections:':<28}{summary['correct_detections']:>18}")
    print(f"{'Detection success rate:':<28}{summary['detection_success_rate']:>17.1f}%")

    print(f"{'-' * width}")

    print(f"{'Average IoU:':<28}{format_average(summary['average_iou']):>18}")
    print(f"{'Average regions examined:':<28}{format_average(summary['average_regions_examined']):>18}")
    print(f"{'Average search time:':<28}{format_average(summary['average_search_time'], ' s'):>18}")
    print(f"{'Average reward:':<28}{format_average(summary['average_reward']):>18}")

    print(f"{'-' * width}\n")