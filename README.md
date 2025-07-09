# YOLO11 Nano Object Detection Website

A real-time object detection website that runs your custom YOLO11 nano model directly in the browser using ONNX Runtime Web.

## Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JavaScript (ES modules)
- **ML Runtime**: ONNX Runtime Web (WebAssembly backend)
- **Model**: YOLOv11-nano (320x320 input, single class detection)
- **Camera**: WebRTC getUserMedia API
- **Hosting**: Static HTTPS (GitHub Pages compatible)


## File Structure

```
Website/
├── index.html          # Main HTML page
├── style.css           # Responsive CSS styling
├── app.js              # Core application logic
├── yolomodel.onnx      # Your trained YOLO model
└── README.md           # This file
```

## Configuration

Edit `app.js` to customize model parameters:

```javascript
const YOLO_CONFIG = {
    modelPath: './yolomodel.onnx',
    inputSize: 320,                    // Model input size
    classes: 1,                        // Number of classes
    confidenceThreshold: 0.5,          // Detection confidence threshold
    iouThreshold: 0.4,                 // NMS IoU threshold
    maxDetections: 100                 // Maximum detections per frame
};
```

## Mobile Usage

1. Open the website in Safari (iOS) or Chrome (Android)
2. Grant camera permissions when prompted
3. The app will automatically use the rear camera
4. Detections will be shown as green bounding boxes with confidence scores

