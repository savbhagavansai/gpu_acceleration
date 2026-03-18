package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.gpu.GpuDelegateFactory
import kotlin.math.exp

/**
 * Hand Detector using Google Play Services TFLite with GPU
 *
 * Input: RGB image (resized to 256×256)
 * Output: Hand bounding boxes and confidence scores
 */
class HandDetectorGPU(private val context: Context) {

    companion object {
        private const val TAG = "HandDetectorGPU"
        private const val MODEL_NAME = "mediapipe_hand-handdetector.tflite"
        private const val INPUT_SIZE = 256
        private const val NUM_ANCHORS = 2944
        private const val DETECTION_THRESHOLD = 0.5f
    }

    private var interpreter: InterpreterApi? = null
    private var anchors: List<Anchor> = emptyList()
    private var isGpuAvailable = false
    private var actualBackend = "UNKNOWN"

    // Image processor - normalize to [0, 1]
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    init {
        anchors = generateAnchors()
        FileLogger.d(TAG, "✓ Generated ${anchors.size} anchors (expected: $NUM_ANCHORS)")
    }

    /**
     * Initialize with GPU support
     */
    fun initialize(): Task<Boolean> {
        FileLogger.section("Initializing Hand Detector (Play Services GPU)")

        return TfLite.initialize(context).continueWith { task ->
            if (!task.isSuccessful) {
                FileLogger.e(TAG, "✗ Play Services TFLite init failed")
                return@continueWith false
            }

            FileLogger.i(TAG, "✓ Play Services TFLite initialized")

            try {
                val gpuDelegateFactory = GpuDelegateFactory()
                isGpuAvailable = gpuDelegateFactory != null
                FileLogger.i(TAG, "✓ GPU delegate available")
            } catch (e: Exception) {
                isGpuAvailable = false
                FileLogger.e(TAG, "GPU delegate not available: ${e.message}")
            }

            // Load model as ByteBuffer
            val modelBuffer = context.assets.openFd(MODEL_NAME).use { fileDescriptor ->
                val inputStream = fileDescriptor.createInputStream()
                val modelBytes = inputStream.readBytes()
                inputStream.close()
                java.nio.ByteBuffer.allocateDirect(modelBytes.size).apply {
                    put(modelBytes)
                    rewind()
                }
            }

            // Try GPU first
            if (isGpuAvailable) {
                FileLogger.d(TAG, "Loading model with GPU delegate...")
                try {
                    val options = InterpreterApi.Options()
                        .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)
                        .addDelegateFactory(GpuDelegateFactory())

                    interpreter = InterpreterApi.create(modelBuffer, options)
                    actualBackend = "GPU"
                    FileLogger.i(TAG, "✓ Hand Detector ready on GPU")
                    return@continueWith true
                } catch (e: Exception) {
                    FileLogger.e(TAG, "GPU loading failed, falling back to CPU")
                    FileLogger.e(TAG, "  Exception: ${e.javaClass.simpleName}: ${e.message}")
                }
            }

            // Fallback to CPU
            FileLogger.d(TAG, "Loading model on CPU...")
            try {
                val options = InterpreterApi.Options()
                    .setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY)

                interpreter = InterpreterApi.create(modelBuffer, options)
                actualBackend = "CPU"
                FileLogger.i(TAG, "✓ Hand Detector ready on CPU")
                true
            } catch (e: Exception) {
                FileLogger.e(TAG, "✗ Hand Detector init failed: ${e.message}")
                false
            }
        }
    }

    /**
     * Detect hand in bitmap
     */
    fun detectHand(bitmap: Bitmap): DetectionResult? {
        try {
            // Preprocess - convert to FLOAT32 and normalize to [0, 1]
            val tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)
            val processedImage = imageProcessor.process(tensorImage)

            // Prepare output buffers
            val outputBoxes = Array(1) { Array(NUM_ANCHORS) { FloatArray(18) } }
            val outputScores = Array(1) { Array(NUM_ANCHORS) { FloatArray(1) } }

            // Output mapping
            val outputs = mapOf(
                0 to outputBoxes,
                1 to outputScores
            )

            // Run inference
            interpreter?.runForMultipleInputsOutputs(
                arrayOf(processedImage.buffer),
                outputs
            )

            // Decode boxes
            val detections = mutableListOf<Detection>()
            for (i in 0 until NUM_ANCHORS) {
                val rawScore = outputScores[0][i][0]
                val score = sigmoid(rawScore)

                if (score > DETECTION_THRESHOLD) {
                    val box = decodeBox(outputBoxes[0][i], anchors[i])
                    detections.add(Detection(box, score))
                }
            }

            if (detections.isEmpty()) {
                return null
            }

            // Apply NMS
            val nmsDetections = nonMaxSuppression(detections, iouThreshold = 0.3f)

            if (nmsDetections.isEmpty()) {
                return null
            }

            // Return best detection
            val best = nmsDetections.maxByOrNull { it.score }!!

            // Convert to image coordinates
            val scaledBox = floatArrayOf(
                best.box[0] * bitmap.width,
                best.box[1] * bitmap.height,
                best.box[2] * bitmap.width,
                best.box[3] * bitmap.height
            )

            return DetectionResult(scaledBox, best.score)

        } catch (e: Exception) {
            FileLogger.e(TAG, "Detection failed: ${e.message}")
            return null
        }
    }

    /**
     * Generate anchors (same as Python code)
     */
    private fun generateAnchors(): List<Anchor> {
        val anchors = mutableListOf<Anchor>()

        // Stride 8: 16×16 grid = 2048 anchors
        for (y in 0 until 16) {
            for (x in 0 until 16) {
                val cx = (x + 0.5f) * 8 / INPUT_SIZE
                val cy = (y + 0.5f) * 8 / INPUT_SIZE
                anchors.add(Anchor(cx, cy, 1.0f, 1.0f))
            }
        }

        // Stride 16: 8×8 grid = 512 anchors (2 per cell)
        for (y in 0 until 8) {
            for (x in 0 until 8) {
                val cx = (x + 0.5f) * 16 / INPUT_SIZE
                val cy = (y + 0.5f) * 16 / INPUT_SIZE
                anchors.add(Anchor(cx, cy, 1.0f, 1.0f))
                anchors.add(Anchor(cx, cy, 1.0f, 1.0f))
            }
        }

        // Stride 32: 4×4 grid = 384 anchors (6 per cell)
        for (y in 0 until 4) {
            for (x in 0 until 4) {
                val cx = (x + 0.5f) * 32 / INPUT_SIZE
                val cy = (y + 0.5f) * 32 / INPUT_SIZE
                repeat(6) {
                    anchors.add(Anchor(cx, cy, 1.0f, 1.0f))
                }
            }
        }

        return anchors
    }

    /**
     * Decode box from model output
     */
    private fun decodeBox(raw: FloatArray, anchor: Anchor): FloatArray {
        val DECODE_SCALE = INPUT_SIZE.toFloat()

        val cx = raw[0] / DECODE_SCALE + anchor.x
        val cy = raw[1] / DECODE_SCALE + anchor.y
        val w = raw[2] / DECODE_SCALE
        val h = raw[3] / DECODE_SCALE

        return floatArrayOf(
            cx - w / 2,  // x_min
            cy - h / 2,  // y_min
            cx + w / 2,  // x_max
            cy + h / 2   // y_max
        )
    }

    /**
     * Non-maximum suppression
     */
    private fun nonMaxSuppression(
        detections: List<Detection>,
        iouThreshold: Float
    ): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val selected = mutableListOf<Detection>()

        for (det in sorted) {
            var shouldSelect = true
            for (sel in selected) {
                if (computeIoU(det.box, sel.box) > iouThreshold) {
                    shouldSelect = false
                    break
                }
            }
            if (shouldSelect) {
                selected.add(det)
            }
        }

        return selected
    }

    /**
     * Compute Intersection over Union
     */
    private fun computeIoU(box1: FloatArray, box2: FloatArray): Float {
        val x1 = maxOf(box1[0], box2[0])
        val y1 = maxOf(box1[1], box2[1])
        val x2 = minOf(box1[2], box2[2])
        val y2 = minOf(box1[3], box2[3])

        if (x2 < x1 || y2 < y1) return 0f

        val intersection = (x2 - x1) * (y2 - y1)
        val area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        val area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        val union = area1 + area2 - intersection

        return intersection / union
    }

    /**
     * Sigmoid activation
     */
    private fun sigmoid(x: Float): Float {
        return 1.0f / (1.0f + exp(-x))
    }

    /**
     * Get backend info
     */
    fun getBackend(): String = actualBackend

    /**
     * Release resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        FileLogger.i(TAG, "✓ Hand Detector closed")
    }
}

/**
 * Anchor for SSD-style detection
 */
data class Anchor(
    val x: Float,
    val y: Float,
    val w: Float,
    val h: Float
)

/**
 * Detection box with confidence
 */
data class Detection(
    val box: FloatArray,  // [x_min, y_min, x_max, y_max] in [0,1]
    val score: Float
)

/**
 * Detection result in image coordinates
 */
data class DetectionResult(
    val box: FloatArray,  // [x_min, y_min, x_max, y_max] in pixels
    val score: Float
)