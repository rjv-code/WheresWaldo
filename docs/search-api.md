# Flask Search API
The Flask search API connects the web interface to the image search controller. It handles image uploads, validates files, and returns search results in JSON format.

The API supports both the pretrained YOLOv8 detector and the trained Q-learning Agent B.

## Endpoint
**POST** `/api/search`

The endpoint accepts image uploads using `multipart/form-data`.
| Field | Required | Description |
|---|---|---|
|`image`|Yes|Image to search for Waldo|
|`method`|No|`q_learning_b` or `yolo`|

The default search method is `q_learning_b`.

- **Supported formats:** JPG, JPEG, PNG, & WEBP
- **Max image size:** 10 MB
- **Image validation:** Pillow
- **Response Format:** JSON

## Example Request
Start the Flask app and run the following command from the project directory:

```powershell
curl.exe -F "image=@pre-trained-model/19_waldo_found.jpg" -F "method=q_learning_b" http://127.0.0.1:5000/api/search
```

To use the original YOLOv8 detector, change `method=q_learning_b` to `method=yolo`.

Replace the examine image path with the path to an existing image on your computer.

## Example Response
```json
{
    "success": true,
    "mock": false,
    "found": true,
    "message": "Waldo may have been found.",
    "detections": [
        {
            "x1": 380.0,
            "x1": 180.0,
            "x1": 418.0,
            "x1": 254.0,
            "confidence": 0.796,
        }
    ],
    "regions_examined": 2,
    "search_method": "q_learning_b",
    "yolo_tiles_processed": 100
}
```

The response contains:
- `found`: Whether a possible Waldo detection was returned
- `detections`: Bounding-box coordinates and confidence scores
- `regions_examined`: Number of grid cells checked by Agent B or tiles processed by the YOLOv8 baseline
- `search_method`: Detection method used for the request
- `yolo_tiles_processed`: Total YOLO inference tiles processed by Agent B's detector, when available
- `mock`: Whether the results are simulated. Real searches return `false`

If Waldo is not detected, `found` is `false` and `detections` is an empty array.

## Error Handling
The API returns JSON error messages when an upload or search fails.

- **400** - Missing, invalid, or unsupported image, or an invalid search method
- **413** - Image or request is too large
- **500** - Unexpected search controller error

Example error response:

```json
{
    "success": false,
    "error": "INVALID_IMAGE",
    "message": "Invalid or corrupted image"
}
```

## Search Controller
The `/api/search` endpoint calls the `search_image(image, method)` function in `app/search_controller.py`.

The function receives a validated image and selects the requested detection method:
- **YOLOv8 baseline:** Calls the original tiled YOLOv8 detector
- **Q-learning Agent B:** Uses YOLOv8 detections and the trained Q-table to search a 4x4 grid

Both methods returned structured detection results that the Flask app sends to the web interface.

Automated tests for the search controller and API are available in `tests/test_search_controller.py` and `tests/test_routes.py`.