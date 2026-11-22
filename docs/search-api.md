# Flask Search API
The Flask search APi connects the web interface to the image search controller. It handles image uploads, validates files, and returns search results in JSON format.

The controller currently returns simulated results. The YOLOv8 detector and Q-learning search strategy will be integrated later.

## Endpoint
**POST** `/api/search`

The endpoint accepts image uploads using `multipart/form-data` with the field name `image`.

- Supported formats: JPG, JPEG, PNG, & WEBP
- Max image size: 10 MB
- Image validation: Pillow
- Response Format: JSON

## Example Request
Start the Flask app and run the following command from the project directory:

```powershell
curl.exe -F "image=@pre-trained-model/19_waldo_found.jpg" http://127.0.0.1:5000/api/search
```

## Example Response
```json
{
    "success": true,
    "mock": true,
    "found": true,
    "message": "Simulated detection, no AI detection",
    "detections": [
        {
            "x1": 100.0,
            "x1": 150.0,
            "x1": 150.0,
            "x1": 230.0,
            "confidence": 0.95,
        }
    ]
}
```

The `detections` array contains possible Waldo locations using bounding-box coordinates and confidence scores.

The `mock` field indicates whether the results are simulated.

## Error Handling
The API returns JSON error messages when an upload or search fails.

- **400** - Missing, invalid, or unsupported image
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
The `/api/search` endpoint calls the `search_image(image)` function in `app/search_controller.py`.

The function receives a validated image and returns structured search results.

During integration, the placeholder will be replaced with our YOLOv8 and Q-learning search pipeline.

Automated testing and performance evaluation will be implemented separately.