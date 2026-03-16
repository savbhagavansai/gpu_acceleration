package com.gesture.recognition

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.gpu.GpuDelegateFactory
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Hand Detector using Google Play Services TFLite with GPU acceleration
 */
class HandDetectorGPU(private val context: Context) {

    companion object {
        private const val TAG = "HandDetectorGPU"
        private const val MODEL_NAME = "mediapipe_hand-handdetector.tflite"
        private const val INPUT_SIZE = 192
        private const val NUM_ANCHORS = 2944
    }

    private var interpreter: InterpreterApi? = null
    private var isGpuAvailable = false
    private var actualBackend = "UNKNOWN"

    // Anchors for hand detection
    private val anchors = generateAnchors()

    // Image processor
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))  // Normalize to [-1, 1]
        .build()

    /**
     * Initialize with GPU support
     * Returns a Task that completes when initialization is done
     */
    fun initialize(): Task<Boolean> {
        FileLogger.section("Initializing Hand Detector (Play Services GPU)")

        return TfLite.initialize(context).continueWithTask { initTask ->
            if (!initTask.isSuccessful) {
                FileLogger.e(TAG, "TfLite initialization failed", initTask.exception)
                return@continueWithTask Tasks.forResult(false)
            }

            FileLogger.i(TAG, "✓ Play Services TFLite initialized")

            // Check GPU availability
            checkGpuAvailability().continueWith { gpuTask ->
                isGpuAvailable = gpuTask.result ?: false

                if (isGpuAvailable) {
                    FileLogger.i(TAG, "✓ GPU delegate available")
                    loadModelWithGpu()
                } else {
                    FileLogger.w(TAG, "GPU not available, using CPU")
                    loadModelWithCpu()
                }

                true
            }
        }
    }

    /**
     * Check if GPU delegate is available
     */
    private fun checkGpuAvailability(): Task<Boolean> {
        return try {
            com.google.android.gms.tflite.gpu.support.TfLiteGpu.isGpuDelegateAvailable(context)
        } catch (e: Exception) {
            FileLogger.w(TAG, "GPU check failed: ${e.message}")
            Tasks.forResult(false)
        }
    }

    /**
     * Load model with GPU acceleration
     */
    private fun loadModelWithGpu() {
        try {
            FileLogger.d(TAG, "Loading model with GPU delegate...")

            val modelBuffer = loadModelFile()

            val options = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_APPLICATION_ONLY)
                .addDelegateFactory(GpuDelegateFactory())

            interpreter = InterpreterApi.create(modelBuffer, options)
            actualBackend = "GPU (Mali-G68 MP5)"

            FileLogger.i(TAG, "✓ Hand Detector ready on GPU")
            Log.d(TAG, "✓ Model loaded on GPU")

        } catch (e: Exception) {
            FileLogger.e(TAG, "GPU loading failed, falling back to CPU", e)
            loadModelWithCpu()
        }
    }

    /**
     * Load model with CPU fallback
     */
    private fun loadModelWithCpu() {
        try {
            FileLogger.d(TAG, "Loading model on CPU...")

            val modelBuffer = loadModelFile()

            val options = InterpreterApi.Options()
                .setRuntime(TfLiteRuntime.FROM_APPLICATION_ONLY)
                .setNumThreads(4)

            interpreter = InterpreterApi.create(modelBuffer, options)
            actualBackend = "CPU (4 threads)"

            FileLogger.i(TAG, "✓ Hand Detector ready on CPU")
            Log.d(TAG, "✓ Model loaded on CPU")

        } catch (e: Exception) {
            FileLogger.e(TAG, "Model loading failed!", e)
            throw e
        }
    }

    /**
     * Load model file from assets
     */
    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = context.assets.openFd(MODEL_NAME)
        val inputStream = java.io.FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(
            java.nio.channels.FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )
    }

    /**
     * Detect hand in image
     */
    fun detectHand(bitmap: Bitmap): DetectionResult? {
        val interp = interpreter ?: run {
            FileLogger.e(TAG, "Interpreter not initialized!")
            return null
        }

        try {
            // Create TensorImage with explicit FLOAT32 type and load bitmap
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)

            // Apply preprocessing (resize + normalize)
            tensorImage = imageProcessor.process(tensorImage)

            // Prepare output buffers
            val outputBoxes = Array(1) { Array(NUM_ANCHORS) { FloatArray(18) } }
            val outputScores = Array(1) { FloatArray(NUM_ANCHORS) }

            val outputs = mapOf(
                0 to outputBoxes,
                1 to outputScores
            )

            // Run inference
            interp.runForMultipleInputsOutputs(
                arrayOf(tensorImage.buffer),
                outputs
            )

            // Process detections
            val detection = processDetections(
                outputBoxes[0],
                outputScores[0],
                bitmap.width,
                bitmap.height
            )

            return detection

        } catch (e: Exception) {
            FileLogger.e(TAG, "Detection failed", e)
            return null
        }
    }

    /**
     * Process raw detections with NMS
     */
    private fun processDetections(
        boxes: Array<FloatArray>,
        scores: FloatArray,
        imageWidth: Int,
        imageHeight: Int
    ): DetectionResult? {

        val detections = mutableListOf<Detection>()

        // Collect detections above threshold
        for (i in scores.indices) {
            val score = sigmoid(scores[i])
            if (score > 0.5f) {
                val anchor = anchors[i]
                val box = decodeBox(boxes[i], anchor, imageWidth, imageHeight)
                detections.add(Detection(box, score))
            }
        }

        if (detections.isEmpty()) return null

        // Apply NMS
        val nmsDetections = applyNMS(detections, 0.3f)

        return nmsDetections.maxByOrNull { it.score }?.let {
            DetectionResult(it.box, it.score)
        }
    }

    /**
     * Generate anchors for SSD detector
     */
    private fun generateAnchors(): List<Anchor> {
        val anchors = mutableListOf<Anchor>()

        // SSD MultiBox anchor generation
        val strides = intArrayOf(8, 16, 16, 16)
        val aspectRatios = listOf(1.0f)

        var layerIdx = 0
        for (stride in strides) {
            val featureMapSize = INPUT_SIZE / stride

            for (y in 0 until featureMapSize) {
                for (x in 0 until featureMapSize) {
                    val xCenter = (x + 0.5f) * stride / INPUT_SIZE
                    val yCenter = (y + 0.5f) * stride / INPUT_SIZE

                    for (ratio in aspectRatios) {
                        val scale = if (layerIdx == 0) 0.1484375f else 0.75f
                        val w = scale
                        val h = scale * ratio

                        anchors.add(Anchor(xCenter, yCenter, w, h))
                    }
                }
            }
            layerIdx++
        }

        FileLogger.d(TAG, "✓ Generated ${anchors.size} anchors")
        return anchors
    }

    /**
     * Decode box from anchor
     */
    private fun decodeBox(
        boxData: FloatArray,
        anchor: Anchor,
        imageWidth: Int,
        imageHeight: Int
    ): FloatArray {
        val cx = boxData[0] / INPUT_SIZE * anchor.w + anchor.x
        val cy = boxData[1] / INPUT_SIZE * anchor.h + anchor.y
        val w = boxData[2] / INPUT_SIZE * anchor.w
        val h = boxData[3] / INPUT_SIZE * anchor.h

        // Convert to pixel coordinates
        val xMin = (cx - w / 2) * imageWidth
        val yMin = (cy - h / 2) * imageHeight
        val xMax = (cx + w / 2) * imageWidth
        val yMax = (cy + h / 2) * imageHeight

        // Extract keypoints (7 hand keypoints)
        val keypoints = FloatArray(14)
        for (i in 0 until 7) {
            keypoints[i * 2] = (boxData[4 + i * 2] / INPUT_SIZE * anchor.w + anchor.x) * imageWidth
            keypoints[i * 2 + 1] = (boxData[5 + i * 2] / INPUT_SIZE * anchor.h + anchor.y) * imageHeight
        }

        return floatArrayOf(xMin, yMin, xMax, yMax, *keypoints)
    }

    /**
     * Apply Non-Maximum Suppression
     */
    private fun applyNMS(detections: List<Detection>, iouThreshold: Float): List<Detection> {
        val sorted = detections.sortedByDescending { it.score }
        val selected = mutableListOf<Detection>()

        for (detection in sorted) {
            var shouldKeep = true

            for (kept in selected) {
                val iou = calculateIoU(detection.box, kept.box)
                if (iou > iouThreshold) {
                    shouldKeep = false
                    break
                }
            }

            if (shouldKeep) {
                selected.add(detection)
            }
        }

        return selected
    }

    /**
     * Calculate Intersection over Union
     */
    private fun calculateIoU(box1: FloatArray, box2: FloatArray): Float {
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
    private fun sigmoid(x: Float): Float = 1.0f / (1.0f + Math.exp(-x.toDouble()).toFloat())

    /**
     * Get backend info
     */
    fun getBackend(): String = actualBackend

    /**
     * Release resources
     */
    fun close() {
        interpreter?.close()
        FileLogger.i(TAG, "✓ Hand Detector closed")
    }
}

/**
 * Data classes
 */
data class Anchor(val x: Float, val y: Float, val w: Float, val h: Float)
data class Detection(val box: FloatArray, val score: Float)
data class DetectionResult(val box: FloatArray, val score: Float)