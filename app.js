// WASM EP (default build)
import * as ort from
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/esm/ort.min.js';

// Configure ONNX Runtime
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
ort.env.wasm.simd = true;
ort.env.wasm.numThreads = 1; // Single-threaded for better mobile compatibility

// YOLO configuration
const YOLO_CONFIG = {
    modelPath: './yolomodel.onnx',
    inputSize: 320,
    classes: 1, // Single class detection
    confidenceThreshold: 0.5,
    iouThreshold: 0.4,
    maxDetections: 100
};

// Global variables
let session = null;
let videoElement = null;
let canvasElement = null;
let ctx = null;
let isRunning = false;
let lastFrameTime = 0;
let frameCount = 0;
let fpsStartTime = Date.now();

// UI elements
const statusElement = document.getElementById('status');
// Removed start/stop buttons – camera starts automatically
const startButton = null;
const stopButton = null;
const fpsDisplay = document.getElementById('fpsDisplay');
const inferenceTimeDisplay = document.getElementById('inferenceTime');

// Initialize the application
async function init() {
    try {
        updateStatus('Loading model...', 'loading');
        
        // Initialize video and canvas elements
        videoElement = document.getElementById('videoElement');
        canvasElement = document.getElementById('overlay');
        ctx = canvasElement.getContext('2d');
        
        // Load ONNX model
        await loadModel();
        
        // Video size listeners (metadata/resize)
        videoElement.addEventListener('loadedmetadata', () => {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
        });
        videoElement.addEventListener('resize', () => {
            canvasElement.width = videoElement.videoWidth;
            canvasElement.height = videoElement.videoHeight;
        });

        // Model loaded – start camera immediately
        updateStatus('Model loaded - Starting camera...', 'loading');
        await startCamera();
        
    } catch (error) {
        console.error('Initialization failed:', error);
        updateStatus('Failed to load model', 'error');
    }
}

// Load ONNX model
async function loadModel() {
    try {
        const sessionOptions = {
            executionProviders: ['wasm'],
            enableCpuMemArena: false,
            enableMemPattern: false,
            enableProfiling: false,
            logSeverityLevel: 3,
            logVerbosityLevel: 0
        };
        
        session = await ort.InferenceSession.create(YOLO_CONFIG.modelPath, sessionOptions);
        console.log('Model loaded successfully');
        console.log('Input names:', session.inputNames);
        console.log('Output names:', session.outputNames);
        
    } catch (error) {
        throw new Error(`Failed to load model: ${error.message}`);
    }
}

// Set up event listeners
function setupEventListeners() {
    // (Buttons removed)
    
    // Handle video metadata loaded
    videoElement.addEventListener('loadedmetadata', () => {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
    });
    
    // Handle video resize
    videoElement.addEventListener('resize', () => {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
    });
}

// Start camera and detection
async function startCamera() {
    try {
        updateStatus('Starting camera...', 'loading');
        
        // Request camera access with rear camera preference
        let stream;
        const preferredConstraints = {
            audio: false,
            video: {
                facingMode: { ideal: 'environment' }, // Prefer rear camera on mobile
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        };

        try {
            stream = await navigator.mediaDevices.getUserMedia(preferredConstraints);
        } catch (err) {
            console.warn('Preferred constraints failed, falling back to default camera', err);
            // Fallback: any available camera
            const fallbackConstraints = { audio: false, video: true };
            stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
        }
        
        videoElement.srcObject = stream;
        // Some browsers (e.g., Safari, iOS) require an explicit play() call
        try {
            await videoElement.play();
        } catch (playErr) {
            console.warn('Autoplay prevented, waiting for user gesture to start video.', playErr);
            // If autoplay is blocked, inform the user to tap the screen to start the camera
            updateStatus('Tap to start camera', 'ready');

            const gestureStart = () => {
                videoElement.play().finally(() => {
                    videoElement.removeEventListener('click', gestureStart);
                    updateStatus('Camera active - Detecting objects', 'ready');
                });
            };
            videoElement.addEventListener('click', gestureStart);
        }
        isRunning = true;
        
        // Buttons removed – nothing to toggle
        
        updateStatus('Camera active - Detecting objects', 'ready');
        
        // Start detection loop
        detectObjects();
        
    } catch (error) {
        console.error('Camera access failed:', error);
        updateStatus('Camera access denied', 'error');
    }
}

// Stop camera and detection
function stopCamera() {
    isRunning = false;
    
    if (videoElement.srcObject) {
        const tracks = videoElement.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Buttons removed – nothing to toggle
    
    updateStatus('Camera stopped', 'ready');
}

// Main detection loop
async function detectObjects() {
    if (!isRunning) return;
    
    try {
        const currentTime = Date.now();
        
        // Process frame
        const inferenceStart = performance.now();
        const detections = await processFrame();
        const inferenceTime = performance.now() - inferenceStart;
        
        // Draw detections
        drawDetections(detections);
        
        // Update performance metrics
        updatePerformanceMetrics(inferenceTime);
        
        // Continue detection loop
        requestAnimationFrame(detectObjects);
        
    } catch (error) {
        console.error('Detection error:', error);
        updateStatus('Detection error', 'error');
    }
}

// Process a single frame
async function processFrame() {
    if (!videoElement.videoWidth || !videoElement.videoHeight) {
        return [];
    }
    
    // Create input tensor from video frame
    const inputTensor = await videoToTensor(videoElement);
    
    // Run inference
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    
    // Process YOLO output
    const detections = processYOLOOutput(output);
    
    // Apply Non-Maximum Suppression
    const filteredDetections = applyNMS(detections);
    
    return filteredDetections;
}

// Convert video frame to tensor
async function videoToTensor(videoElement) {
    // Create canvas for image processing
    const canvas = document.createElement('canvas');
    canvas.width = YOLO_CONFIG.inputSize;
    canvas.height = YOLO_CONFIG.inputSize;
    const ctx = canvas.getContext('2d');
    
    // Draw video frame to canvas (resized)
    ctx.drawImage(videoElement, 0, 0, YOLO_CONFIG.inputSize, YOLO_CONFIG.inputSize);
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, YOLO_CONFIG.inputSize, YOLO_CONFIG.inputSize);
    const data = imageData.data;
    
    // Convert to RGB and normalize
    const tensorData = new Float32Array(3 * YOLO_CONFIG.inputSize * YOLO_CONFIG.inputSize);
    
    for (let i = 0; i < YOLO_CONFIG.inputSize * YOLO_CONFIG.inputSize; i++) {
        const pixelIndex = i * 4; // RGBA
        
        // Normalize to [0, 1]
        tensorData[i] = data[pixelIndex] / 255.0; // R
        tensorData[i + YOLO_CONFIG.inputSize * YOLO_CONFIG.inputSize] = data[pixelIndex + 1] / 255.0; // G
        tensorData[i + 2 * YOLO_CONFIG.inputSize * YOLO_CONFIG.inputSize] = data[pixelIndex + 2] / 255.0; // B
    }
    
    return new ort.Tensor('float32', tensorData, [1, 3, YOLO_CONFIG.inputSize, YOLO_CONFIG.inputSize]);
}

// Process YOLO output
function processYOLOOutput(output) {
    const detections = [];
    const data = output.data;
    const dims = output.dims;
    
    // YOLOv11 output format: [batch, 84, 8400] where 84 = 4 (bbox) + 80 (classes)
    // For single class: [batch, 5, num_anchors] where 5 = 4 (bbox) + 1 (class)
    const numDetections = dims[2];
    
    for (let i = 0; i < numDetections; i++) {
        // Extract bounding box (center_x, center_y, width, height)
        const centerX = data[i * 5 + 0];
        const centerY = data[i * 5 + 1];
        const width = data[i * 5 + 2];
        const height = data[i * 5 + 3];
        const confidence = data[i * 5 + 4];
        
        // Filter by confidence
        if (confidence > YOLO_CONFIG.confidenceThreshold) {
            // Convert to corner coordinates
            const x1 = centerX - width / 2;
            const y1 = centerY - height / 2;
            const x2 = centerX + width / 2;
            const y2 = centerY + height / 2;
            
            detections.push({
                x1: Math.max(0, x1),
                y1: Math.max(0, y1),
                x2: Math.min(1, x2),
                y2: Math.min(1, y2),
                confidence: confidence,
                class: 0 // Single class
            });
        }
    }
    
    return detections;
}

// Apply Non-Maximum Suppression
function applyNMS(detections) {
    if (detections.length === 0) return [];
    
    // Sort by confidence (descending)
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const keep = [];
    const suppressed = new Set();
    
    for (let i = 0; i < detections.length; i++) {
        if (suppressed.has(i)) continue;
        
        keep.push(detections[i]);
        
        // Suppress overlapping detections
        for (let j = i + 1; j < detections.length; j++) {
            if (suppressed.has(j)) continue;
            
            const iou = calculateIOU(detections[i], detections[j]);
            if (iou > YOLO_CONFIG.iouThreshold) {
                suppressed.add(j);
            }
        }
    }
    
    return keep.slice(0, YOLO_CONFIG.maxDetections);
}

// Calculate Intersection over Union
function calculateIOU(box1, box2) {
    const x1 = Math.max(box1.x1, box2.x1);
    const y1 = Math.max(box1.y1, box2.y1);
    const x2 = Math.min(box1.x2, box2.x2);
    const y2 = Math.min(box1.y2, box2.y2);
    
    if (x2 <= x1 || y2 <= y1) return 0;
    
    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    const area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    const union = area1 + area2 - intersection;
    
    return intersection / union;
}

// Draw detections on canvas
function drawDetections(detections) {
    // Clear canvas
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Draw each detection
    detections.forEach(detection => {
        const videoWidth = videoElement.videoWidth;
        const videoHeight = videoElement.videoHeight;
        
        // Convert normalized coordinates to pixel coordinates
        const x1 = detection.x1 * videoWidth;
        const y1 = detection.y1 * videoHeight;
        const x2 = detection.x2 * videoWidth;
        const y2 = detection.y2 * videoHeight;
        
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Draw bounding box
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw confidence label
        const confidence = Math.round(detection.confidence * 100);
        const label = `${confidence}%`;
        
        ctx.fillStyle = '#00ff00';
        ctx.font = '16px Arial';
        ctx.fillText(label, x1, y1 - 5);
    });
}

// Update performance metrics
function updatePerformanceMetrics(inferenceTime) {
    frameCount++;
    const currentTime = Date.now();
    
    // Update FPS every second
    if (currentTime - fpsStartTime >= 1000) {
        const fps = Math.round(frameCount / ((currentTime - fpsStartTime) / 1000));
        fpsDisplay.textContent = fps;
        frameCount = 0;
        fpsStartTime = currentTime;
    }
    
    // Update inference time
    inferenceTimeDisplay.textContent = Math.round(inferenceTime);
}

// Update status message
function updateStatus(message, type = '') {
    statusElement.textContent = message;
    statusElement.className = `status ${type}`;
}

// Initialize app when DOM is loaded (handles both fresh load and cached navigation)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    // DOM was already loaded (e.g., bfcache restore)
    init();
} 