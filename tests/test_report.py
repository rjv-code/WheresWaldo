import pytest

from evaluation.report import summarize_results, print_report


def test_summarize_results():
    """Checks calculations for multiple evaluation results"""
    results = [
        {
            "classification": "correct",
            "iou": 0.8,
            "waldo_present": True,
            "regions_examined": 5,
            "search_time_seconds": 1.0,
            "reward": 1.0
        },
        {
            "classification": "partial",
            "iou": 0.3,
            "waldo_present": True,
            "regions_examined": 9,
            "search_time_seconds": 2.0,
            "reward": 0.5
        },
        {
            "classification": "correct",
            "iou": None,
            "waldo_present": False,
            "regions_examined": None,
            "search_time_seconds": 1.5,
            "reward": 1.0
        }
    ]

    summary = summarize_results(results)

    assert summary["total_images"] == 3
    assert summary["correct_detections"] == 2
    assert summary["detection_success_rate"] == pytest.approx(66.67, rel=0.001)
    assert summary["average_iou"] == pytest.approx(0.55)
    assert summary["average_regions_examined"] == pytest.approx(7.0)
    assert summary["average_search_time"] == pytest.approx(1.5)
    assert summary["average_reward"] == pytest.approx(2.5 / 3)


def test_empty_results():
    """Checks that an empty evaluation produces valid summary values"""
    summary = summarize_results([])

    assert summary["total_images"] == 0
    assert summary["correct_detections"] == 0
    assert summary["detection_success_rate"] == 0.0
    assert summary["average_iou"] is None
    assert summary["average_regions_examined"] is None
    assert summary["average_search_time"] is None
    assert summary["average_reward"] is None


def test_print_report(capsys):
    """Checks that the evaluation summary is printed to the console"""
    summary = {
        "total_images": 10,
        "correct_detections": 8,
        "detection_success_rate": 80.0,
        "average_iou": 0.75,
        "average_regions_examined": 6.5,
        "average_search_time": 1.25,
        "average_reward": 0.7
    }

    print_report(summary)

    output = capsys.readouterr().out

    assert "Finding Waldo - Evaluation Results" in output
    assert "Images evaluated:" in output
    assert "Detection success rate:" in output
    assert "80.0%" in output
    assert "Average IoU:" in output
    assert "Average regions examined:" in output
    assert "Average search time:" in output


def test_print_empty_report(capsys):
    """Checks that missing averages are displayed as N/A"""
    print_report(summarize_results([]))

    output = capsys.readouterr().out

    assert "N/A" in output
