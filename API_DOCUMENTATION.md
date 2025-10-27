# API Documentation

Complete REST API documentation for the Pneumonia Detection System.

## Base URL

```
http://localhost:5000
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

---

## Endpoints

### 1. Health Check

Check if the API server is running and the model is loaded.

**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded"
}
```

**Status Codes:**
- `200 OK`: Server is healthy
- `500 Internal Server Error`: Server error

**Example:**
```bash
curl http://localhost:5000/api/health
```

---

### 2. Model Information

Get information about the loaded model.

**Endpoint:** `GET /api/model-info`

**Response:**
```json
{
  "model_name": "Pneumonia Detection Model",
  "input_shape": [224, 224, 3],
  "classes": ["Normal", "Pneumonia"],
  "model_path": "models/pneumonia_model.h5"
}
```

**Status Codes:**
- `200 OK`: Success
- `500 Internal Server Error`: Model not loaded

**Example:**
```bash
curl http://localhost:5000/api/model-info
```

---

### 3. Simple Prediction

Make a prediction on an uploaded X-ray image without Grad-CAM visualization.

**Endpoint:** `POST /api/predict`

**Content-Type:** `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | X-ray image (PNG, JPG, JPEG) |

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Pneumonia",
    "confidence": 0.9523,
    "probability": 0.9523,
    "is_pneumonia": true
  }
}
```

**Response Fields:**
- `success` (boolean): Whether the prediction was successful
- `prediction.class` (string): Predicted class ("Normal" or "Pneumonia")
- `prediction.confidence` (float): Confidence score (0-1)
- `prediction.probability` (float): Raw probability of pneumonia class
- `prediction.is_pneumonia` (boolean): True if pneumonia detected

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: No file provided or invalid file type
- `500 Internal Server Error`: Prediction error

**Example (cURL):**
```bash
curl -X POST \
  http://localhost:5000/api/predict \
  -F "file=@path/to/xray.jpg"
```

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/predict"
files = {"file": open("xray.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Example (JavaScript):**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

---

### 4. Prediction with Grad-CAM

Make a prediction with Grad-CAM visualization showing which regions influenced the decision.

**Endpoint:** `POST /api/predict-with-gradcam`

**Content-Type:** `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | X-ray image (PNG, JPG, JPEG) |

**Response:**
```json
{
  "success": true,
  "prediction": {
    "class": "Pneumonia",
    "confidence": 0.9523,
    "probability": 0.9523,
    "is_pneumonia": true
  },
  "gradcam": {
    "original": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "heatmap": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
    "overlay": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
  }
}
```

**Response Fields:**
- `success` (boolean): Whether the prediction was successful
- `prediction` (object): Same as simple prediction
- `gradcam.original` (string): Base64-encoded original X-ray image
- `gradcam.heatmap` (string): Base64-encoded Grad-CAM heatmap
- `gradcam.overlay` (string): Base64-encoded overlay visualization

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: No file provided or invalid file type
- `500 Internal Server Error`: Prediction error

**Example (cURL):**
```bash
curl -X POST \
  http://localhost:5000/api/predict-with-gradcam \
  -F "file=@path/to/xray.jpg"
```

**Example (Python):**
```python
import requests
import base64
from PIL import Image
import io

url = "http://localhost:5000/api/predict-with-gradcam"
files = {"file": open("xray.jpg", "rb")}
response = requests.post(url, files=files)
data = response.json()

# Decode and save overlay image
overlay_data = data['gradcam']['overlay'].split(',')[1]
overlay_bytes = base64.b64decode(overlay_data)
overlay_image = Image.open(io.BytesIO(overlay_bytes))
overlay_image.save('gradcam_result.png')
```

---

### 5. Batch Prediction

Make predictions on multiple images at once.

**Endpoint:** `POST /api/batch-predict`

**Content-Type:** `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| files | File[] | Yes | Multiple X-ray images |

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "filename": "image1.jpg",
      "prediction": {
        "class": "Normal",
        "confidence": 0.8234,
        "probability": 0.1766,
        "is_pneumonia": false
      }
    },
    {
      "filename": "image2.jpg",
      "prediction": {
        "class": "Pneumonia",
        "confidence": 0.9523,
        "probability": 0.9523,
        "is_pneumonia": true
      }
    }
  ],
  "total_processed": 2
}
```

**Status Codes:**
- `200 OK`: Batch prediction successful
- `400 Bad Request`: No files provided
- `500 Internal Server Error`: Prediction error

**Example (Python):**
```python
import requests

url = "http://localhost:5000/api/batch-predict"
files = [
    ('files', open('xray1.jpg', 'rb')),
    ('files', open('xray2.jpg', 'rb')),
    ('files', open('xray3.jpg', 'rb'))
]
response = requests.post(url, files=files)
print(response.json())
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "No file provided"
}
```

### 413 Request Entity Too Large
```json
{
  "error": "File too large. Maximum size is 16MB"
}
```

### 500 Internal Server Error
```json
{
  "error": "Model not available"
}
```

---

## Rate Limiting

Currently, there are no rate limits. For production:
- Implement rate limiting (e.g., 100 requests/hour per IP)
- Use Redis for distributed rate limiting
- Consider API key-based quotas

---

## File Upload Constraints

- **Allowed formats**: PNG, JPG, JPEG
- **Maximum file size**: 16 MB
- **Recommended image size**: 224×224 to 2048×2048 pixels
- **Color mode**: RGB or Grayscale (automatically converted)

---

## Response Times

Typical response times:
- Simple prediction: 200-500ms
- Prediction with Grad-CAM: 500-1000ms
- Batch prediction (10 images): 2-5 seconds

*Times may vary based on hardware and image size*

---

## CORS Configuration

CORS is enabled for all origins in development. For production:

```python
# app.py
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://yourdomain.com"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

---

## Webhooks (Future Feature)

Planned webhook support for asynchronous processing:

```json
POST /api/predict-async
{
  "image_url": "https://example.com/xray.jpg",
  "callback_url": "https://yourapp.com/webhook"
}
```

---

## SDK Examples

### Python SDK

```python
class PneumoniaDetectionClient:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
    
    def predict(self, image_path):
        url = f"{self.base_url}/api/predict"
        files = {"file": open(image_path, "rb")}
        response = requests.post(url, files=files)
        return response.json()
    
    def predict_with_gradcam(self, image_path):
        url = f"{self.base_url}/api/predict-with-gradcam"
        files = {"file": open(image_path, "rb")}
        response = requests.post(url, files=files)
        return response.json()

# Usage
client = PneumoniaDetectionClient()
result = client.predict("xray.jpg")
print(f"Prediction: {result['prediction']['class']}")
```

### JavaScript SDK

```javascript
class PneumoniaDetectionClient {
  constructor(baseUrl = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }
  
  async predict(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      body: formData
    });
    
    return await response.json();
  }
  
  async predictWithGradCAM(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${this.baseUrl}/api/predict-with-gradcam`, {
      method: 'POST',
      body: formData
    });
    
    return await response.json();
  }
}

// Usage
const client = new PneumoniaDetectionClient();
const result = await client.predict(fileInput.files[0]);
console.log(`Prediction: ${result.prediction.class}`);
```

---

## Testing the API

### Using Postman

1. Create a new POST request
2. Set URL to `http://localhost:5000/api/predict-with-gradcam`
3. Go to Body → form-data
4. Add key "file" with type "File"
5. Select an X-ray image
6. Send request

### Using cURL

```bash
# Simple prediction
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:5000/api/predict

# With Grad-CAM
curl -X POST \
  -F "file=@test_image.jpg" \
  http://localhost:5000/api/predict-with-gradcam \
  -o response.json
```

---

## Production Considerations

### Security
- [ ] Implement authentication (JWT/API keys)
- [ ] Add rate limiting
- [ ] Validate file types server-side
- [ ] Scan uploads for malware
- [ ] Use HTTPS only
- [ ] Implement CORS properly

### Performance
- [ ] Add caching for repeated requests
- [ ] Implement request queuing
- [ ] Use async processing for batch requests
- [ ] Add CDN for static assets
- [ ] Optimize model loading

### Monitoring
- [ ] Log all API requests
- [ ] Track prediction accuracy
- [ ] Monitor response times
- [ ] Set up error alerting
- [ ] Track API usage metrics

---

## Support

For API issues or questions:
- GitHub Issues: [repository-url]/issues
- Email: support@example.com
- Documentation: [repository-url]/wiki

---

**Last Updated**: 2024
**API Version**: 1.0.0
