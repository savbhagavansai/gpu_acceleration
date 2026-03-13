package com.gesture.recognition

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.view.Surface
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.Exception

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }

    // UI Components
    private lateinit var viewFinder: PreviewView
    private lateinit var gestureOverlay: GestureOverlayView
    private lateinit var statusText: TextView
    private lateinit var fpsText: TextView
    private lateinit var backendText: TextView

    // Camera
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var preview: Preview? = null
    private var camera: Camera? = null
    private lateinit var cameraExecutor: ExecutorService

    // Gesture Recognition
    private var gestureRecognizer: GestureRecognizerGPU? = null
    private var isProcessing = false
    private var frameCount = 0
    private var lastFpsTime = System.currentTimeMillis()
    private var currentFps = 0.0

    // Frame skipping for performance
    private var frameSkipCounter = 0
    private val FRAME_SKIP = 1  // Process every 2nd frame

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set up crash handler
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            FileLogger.e(TAG, "═══ APP CRASHED ═══")
            FileLogger.e(TAG, "  Exception: ${throwable.javaClass.simpleName}: ${throwable.message}")
            throwable.stackTrace.take(5).forEach { element ->
                FileLogger.e(TAG, "    at ${element}")
            }
            FileLogger.section("APP STOPPED: ${FileLogger.getCurrentTimestamp()}")
        }

        setContentView(R.layout.activity_main)

        // Initialize FileLogger
        FileLogger.initialize(this)
        FileLogger.section("APP STARTED: ${FileLogger.getCurrentTimestamp()}")
        FileLogger.section("Log file: ${FileLogger.getLogFilePath()}")

        // Initialize UI
        initializeUI()

        // Log device info
        logDeviceInfo()

        // Check permissions
        if (allPermissionsGranted()) {
            initializeApp()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    /**
     * Initialize UI components
     */
    private fun initializeUI() {
        viewFinder = findViewById(R.id.viewFinder)
        gestureOverlay = findViewById(R.id.gestureOverlay)
        statusText = findViewById(R.id.statusText)
        fpsText = findViewById(R.id.fpsText)
        backendText = findViewById(R.id.backendText)

        // Set initial text
        statusText.text = "Initializing..."
        fpsText.text = "FPS: --"
        backendText.text = "Backend: --"
    }

    /**
     * Log device information
     */
    private fun logDeviceInfo() {
        FileLogger.section("MainActivity onCreate()")
        FileLogger.i(TAG, "App started - processing at 640x480")
        FileLogger.i(TAG, "Android version: ${android.os.Build.VERSION.SDK_INT}")
        FileLogger.i(TAG, "Device: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}")
        FileLogger.i(TAG, "Log location: Downloads/GestureRecognition/debug_log.txt")

        // Check model files
        checkModelFiles()
    }

    /**
     * Check if model files exist
     */
    private fun checkModelFiles() {
        try {
            val handDetectorSize = assets.openFd("mediapipe_hand-handdetector.tflite").length / 1024
            val handLandmarkSize = assets.openFd("mediapipe_hand-handlandmarkdetector.tflite").length / 1024
            val gestureSize = assets.openFd("gesture_model.onnx").length / 1024

            FileLogger.i(TAG, "✓ Model files found:")
            FileLogger.i(TAG, "  - HandDetector: ${handDetectorSize}KB")
            FileLogger.i(TAG, "  - HandLandmark: ${handLandmarkSize}KB")
            FileLogger.i(TAG, "  - Gesture: ${gestureSize}KB")
        } catch (e: Exception) {
            FileLogger.e(TAG, "✗ Model files missing!", e)
            showError("Model files not found in assets!")
        }
    }

    /**
     * Initialize the app (called after permissions granted)
     */
    private fun initializeApp() {
        FileLogger.section("Initializing Application")

        // Initialize gesture recognizer with GPU
        initializeGestureRecognizer()
    }

    /**
     * Initialize gesture recognizer with GPU support
     * This is ASYNC because GPU initialization takes time
     */
    private fun initializeGestureRecognizer() {
        try {
            FileLogger.section("Initializing Gesture Recognizer")

            runOnUiThread {
                statusText.text = "Loading AI models..."
            }

            gestureRecognizer = GestureRecognizerGPU(this)

            // Initialize asynchronously
            gestureRecognizer?.initialize()?.addOnCompleteListener { task ->
                if (task.isSuccessful && task.result == true) {
                    FileLogger.i(TAG, "✓ Gesture Recognizer initialized successfully")

                    // Log backend info
                    val detectorBackend = gestureRecognizer?.getDetectorBackend() ?: "UNKNOWN"
                    val landmarkBackend = gestureRecognizer?.getLandmarkBackend() ?: "UNKNOWN"

                    FileLogger.i(TAG, "Backend configuration:")
                    FileLogger.i(TAG, "  - Hand Detector: $detectorBackend")
                    FileLogger.i(TAG, "  - Landmarks: $landmarkBackend")
                    FileLogger.i(TAG, "  - Gesture: NPU (ONNX Runtime)")

                    runOnUiThread {
                        statusText.text = "AI models loaded"
                        backendText.text = "GPU: Mali-G68"

                        // NOW safe to start camera
                        startCamera()
                    }

                } else {
                    val error = task.exception?.message ?: "Unknown error"
                    FileLogger.e(TAG, "Gesture Recognizer initialization failed: $error")

                    runOnUiThread {
                        statusText.text = "Initialization failed"
                        showError("Failed to initialize AI models: $error")
                    }
                }
            }

        } catch (e: Exception) {
            FileLogger.e(TAG, "Failed to create recognizer", e)
            runOnUiThread {
                showError("Initialization error: ${e.message}")
            }
        }
    }

    /**
     * Start camera
     */
    private fun startCamera() {
        FileLogger.i(TAG, "Starting camera...")

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()

                FileLogger.i(TAG, "✓ Camera started successfully")

                runOnUiThread {
                    statusText.text = "Ready"
                }

            } catch (e: Exception) {
                FileLogger.e(TAG, "Camera start failed", e)
                runOnUiThread {
                    showError("Camera error: ${e.message}")
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Bind camera use cases
     */
    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed")

        // Preview use case
        preview = Preview.Builder()
            .build()
            .also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

        // Image analysis use case
        imageAnalyzer = ImageAnalysis.Builder()
            .setTargetResolution(android.util.Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImage(imageProxy)
                }
            }

        // Select front camera
        val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

        try {
            // Unbind all before rebinding
            cameraProvider.unbindAll()

            // Bind use cases to camera
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

        } catch (e: Exception) {
            FileLogger.e(TAG, "Camera binding failed", e)
        }
    }

    /**
     * Process camera image
     */
    private fun processImage(imageProxy: ImageProxy) {
        // Skip frames for performance
        frameSkipCounter++
        if (frameSkipCounter % (FRAME_SKIP + 1) != 0) {
            imageProxy.close()
            return
        }

        // Don't process if already processing
        if (isProcessing) {
            imageProxy.close()
            return
        }

        isProcessing = true
        frameCount++

        try {
            // Log every 30th frame
            if (frameCount % 30 == 1) {
                FileLogger.d(TAG, "────────────────────────────────────────────────────────────")
                FileLogger.d(TAG, "Frame #$frameCount: ${imageProxy.width}x${imageProxy.height}, rotation=${imageProxy.imageInfo.rotationDegrees}")
            }

            // Convert ImageProxy to Bitmap
            val bitmap = imageProxyToBitmap(imageProxy)

            // Recognize gesture
            val result = gestureRecognizer?.recognize(bitmap)

            // Update UI and log
            result?.let { updateUI(it) }

            // Update FPS
            updateFPS()

        } catch (e: Exception) {
            FileLogger.e(TAG, "Frame processing error", e)
        } finally {
            isProcessing = false
            imageProxy.close()
        }
    }

    /**
     * Convert ImageProxy to Bitmap
     */
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)

        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 100, out)
        val imageBytes = out.toByteArray()

        val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

        // Rotate bitmap based on image rotation
        return rotateBitmap(bitmap, imageProxy.imageInfo.rotationDegrees.toFloat())
    }

    /**
     * Rotate bitmap
     */
    private fun rotateBitmap(bitmap: Bitmap, rotation: Float): Bitmap {
        val matrix = Matrix()
        matrix.postRotate(rotation)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Update UI with recognition result
     */
    private fun updateUI(result: GestureResult) {
        runOnUiThread {
            // Update gesture overlay
            result.landmarks?.let { landmarks ->
                gestureOverlay.updateLandmarks(landmarks, 1088, 1088)  // Use actual camera resolution
            }

            // Update status text
            val gesture = result.gesture
            val confidence = (result.confidence * 100).toInt()

            val statusMessage = when {
                gesture == "no_hand" -> "No hand detected"
                gesture == "no_landmarks" -> "Hand tracking lost"
                gesture == "buffering" -> "Buffering... ${result.bufferFilled}/15"
                else -> "$gesture (${confidence}%)"
            }

            statusText.text = statusMessage

            // Log performance every 30th frame
            if (frameCount % 30 == 1) {
                FileLogger.d(TAG, "Result received:")
                FileLogger.d(TAG, "  - Landmarks: ${result.landmarks?.size ?: 0} values")
                FileLogger.d(TAG, "  - Gesture: ${result.gesture}")
                FileLogger.d(TAG, "  - Confidence: ${result.confidence}")
                FileLogger.d(TAG, "  - Buffer: ${result.bufferFilled}/15")
                FileLogger.d(TAG, "  - Detector: ${String.format("%.1f", result.detectorTimeMs)}ms")
                FileLogger.d(TAG, "  - Landmark: ${String.format("%.1f", result.landmarkTimeMs)}ms")
                FileLogger.d(TAG, "  - Total: ${String.format("%.1f", result.totalTimeMs)}ms")
                FileLogger.d(TAG, "  - Was Tracking: ${result.wasTracking}")
            }
        }
    }

    /**
     * Update FPS counter
     */
    private fun updateFPS() {
        val currentTime = System.currentTimeMillis()
        val elapsed = currentTime - lastFpsTime

        if (elapsed >= 1000) {  // Update every second
            currentFps = frameCount * 1000.0 / elapsed
            frameCount = 0
            lastFpsTime = currentTime

            runOnUiThread {
                fpsText.text = String.format("FPS: %.1f", currentFps)
            }
        }
    }

    /**
     * Show error message
     */
    private fun showError(message: String) {
        runOnUiThread {
            Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            statusText.text = "Error"
        }
    }

    /**
     * Check if all permissions are granted
     */
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    /**
     * Handle permission request result
     */
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                initializeApp()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    /**
     * Clean up resources
     */
    override fun onDestroy() {
        super.onDestroy()

        FileLogger.section("MainActivity onDestroy()")

        // Log final stats
        FileLogger.i(TAG, "App closing - processed $frameCount frames")

        // Close gesture recognizer
        gestureRecognizer?.close()

        // Shutdown camera executor
        cameraExecutor.shutdown()

        FileLogger.section("APP STOPPED: ${FileLogger.getCurrentTimestamp()}")
    }
}