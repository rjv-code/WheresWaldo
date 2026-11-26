# Testing and evaluation
Our project uses 

## Automated testing
We use pytest to test the Flask app, image uploads, search controller, and evaluation functions.

Run all tests from the project directory:

```powershell
python -m pytest -v
```

## Evaluation metrics
We measure performance using:
- Detection success rate: Percentage of correct predictions
- Average IoU: How closely predicted locations match Waldo's actual location
- Average regions examined: Number of grid cells checked during a search
- Average search time: Time taken to complete a search
- Average reward: Score assigned to a prediction based on its accuracy

## Reward system
The evaluation framework uses the following rewards:
| Results | Reward |
| --- | --- |
| IoU >= 0.50 | +1 |
| IoU >= 0.25 and < 0.50 | +0.5 |
| IoU < 0.25 | -1 |
| Correctly identifies Waldo as absent | +1 |
| Incorrectly predicts Waldo's presence or absence | -1 |

The Q-learning agent uses a separate training reward system: +10 for correctly finding Waldo, -5 for an incorrect declaration or reaching the step limit, and -1 per move.

## Evaluation results
We compared six search methods using 15 validation and test images containing Waldo.
| Search method | Waldo found | Average cells examined |
|---|---|---|
| Full scan | 12/15 (80%) | 16.0 |
| Row-by-row search | 11/15 (73%) | 9.6 |
| Highest stripe score first | 10/15 (67%) | 8.5 |
| Q-learning Agent B (30,000 episodes) | 9/15 (60%) | 5.1 |
| Q-learning Agent B (100,000 episodes) | 8/15 (53%) | 9.7 |
| Q-learning Agent A (1,500 episodes) | 9/15 (60%) | 5.5 |

The 30,000-episode Agent B policy checked fewer grid cells than the baseline methods but found Waldo less often.

These results show a trade-off between detection accuracy and the number of grid cells examined. They do not establish faster overall inference because YOLOv8 processes the iamge before the Q-learning agent selects regions.

The evaluation also has limitations. Only 15 images containing Waldo were evaluated, and the pretrained detector may have encountered some of theimages during its original training.

## Evaluation files
| File | Purpose |
| --- | --- |
| `evaluation/metrics.py` | Calculates IoU and evaluation rewards |
| `evaluation/runner.py` | Evaluates predictions and records search times |
| `evaluation/report.py` | Summarizes results and displays them in the console |
| `pre-trained-model/rl_tile_search.py` | Trains and evaluates the Q-learning search policy |
| `pre-trained-mode/rl_results_B.json` | Stores the 30,000-episode Agent B results, where available |

## Current progress
The evaluation framework, Q-learning training, and search-method comparisons have been implemented.

The Flask app supports both the YOLOv8 baseline and Q-learning Agent B. Automated tests and browser testing have been used to verify the integrated search workflow.