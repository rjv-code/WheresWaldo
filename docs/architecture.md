# Application architecture
Finding Waldo uses Flask to receive images and JavaScript to show search results. Users can choose between two detection methods.

## Search process
1. The browser sends an image and a detection method to `/api/search`
2. Flask checks the file type and size
3. The search controller starts the selected detector
4. The browser displays possible matches and draws a box around a returned location

## Detection methods
**YOLOv8 baseline** (`app/detector.py`) runs the pretrained model on overlapping 512-px image tiles. It returns bounding boxes and confidence scores.

**Q-learning Agent B** (`app/rl_detector.py`) runs a pretrained YOLOv8 model on 256-px tiles, then uses a 4x4 grid, red-and-white stripe scores, and the saved Q-table to choose which cells to examine. It stops when it accepts a sufficiently confidence match or reaches its step limit.

The policy is trained separately. Searching an uploaded image does not change the Q-table.

**Important:** Agent B does not avoid the full YOLO scan. Its count of *grid cells examined* is different from the number of *YOLO tiles processed*.

## Main files
| File | Purpose |
| --- | --- |
| `app/routes.py` | Flask pages and API endpoints |
| `app/search_controller.py` | Choose the detection method and format results |
| `app/detector.py` | YOLOv8 baseline |
| `app/rl_detector.py` | Agent B inference |
| `app/static/js/main.js` | Chat interface and image annotations |
| `pre-trained-model/rl_tile_search.py` | Q-learning training and experiments |
| `pre-trained-model/q_table_B.npy` | Saved Agent B policy |

For request and response details, see [Search API](search-api.md).