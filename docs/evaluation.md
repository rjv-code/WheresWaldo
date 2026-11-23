# Testing and Evaluation
Our project uses automated tests and evaluation metrics to measure detection accuracy and search efficiency.

## Automated Testing
We use pytest to test the Flask app, image uploads, search controller, and evaluation functions.

Run all tests from the project directory:

```powershell
python -m pytest -v
```

# Evaluation Metrics
We measure the agent's performance using:
- Detection success rate: Percentage of correct predictions
- Average IoU: How closely predicted locations match Waldo's actual location
- Average regions examined: Number of image regions searched
- Average search time: Time taken to complete a search
- Average reward: Score based on correct, partially correct, or incorrect predictions

## Reward System
| Results | Reward |
| --- | --- |
| IoU >= 0.50 | +1 |
| IoU >= 0.25 and < 0.50 | +0.5 |
| IoU < 0.25 | -1 |
| Correctly identifies Waldo as absent | +1 |
| Incorrectly predicts Waldo's presence or absence | -1 |

## Evaluation Files
- `evaluation/metrics.py` - Calculates IoU and rewards
- `evaluation/runner.py` - Evaluates predictions and records search time
- `evaluation/report.py` - Summarizes results and displays them in the console

## Current Progress
The evaluation framework is implemented and can be tested using sample results.

Actual YOLOv8 and Q-learning performance comparisons will be conducted once the AI components are integrated.