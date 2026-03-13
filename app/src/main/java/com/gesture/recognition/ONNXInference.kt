package com.gesture.recognition

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.nio.FloatBuffer

/**
 * ONNX Runtime inference engine
 * Loads INT8 quantized TCN model and performs gesture classification
 */
class ONNXInference(context: Context) {
    
    private val TAG = "ONNXInference"
    
    private val ortEnvironment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var ortSession: OrtSession? = null
    
    private var inputName: String? = null
    private var outputName: String? = null
    
    init {
        loadModel(context)
    }
    
    /**
     * Load ONNX model from assets
     */
    private fun loadModel(context: Context) {
        try {
            Log.d(TAG, "Loading ONNX model: ${Config.ONNX_MODEL_FILENAME}")
            
            // Strategy 1: Try loading from external storage first (better for external data)
            val modelFile = copyAssetToFile(context, Config.ONNX_MODEL_FILENAME)

            // Check for external data file
            val dataFileName = "${Config.ONNX_MODEL_FILENAME}.data"
            var dataFile: java.io.File? = null

            try {
                context.assets.list("")?.let { assetList ->
                    if (assetList.contains(dataFileName)) {
                        Log.d(TAG, "Found external data file: $dataFileName")
                        dataFile = copyAssetToFile(context, dataFileName)
                    }
                }
            } catch (e: Exception) {
                Log.d(TAG, "No external data file")
            }

            Log.d(TAG, "Model file: ${modelFile.absolutePath} (${modelFile.length()} bytes)")
            if (dataFile != null) {
                Log.d(TAG, "Data file: ${dataFile?.absolutePath} (${dataFile?.length()} bytes)")
            }

            // Create session options with optimization
            val sessionOptions = OrtSession.SessionOptions()
            sessionOptions.setIntraOpNumThreads(2)
            sessionOptions.setInterOpNumThreads(2)

            // Try to enable NNAPI (Android Neural Networks API) for hardware acceleration
            try {
                sessionOptions.addNnapi()
                Log.d(TAG, "✓ NNAPI hardware acceleration enabled")
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI not available, using CPU: ${e.message}")
            }

            // Try loading the model
            Log.d(TAG, "Creating ONNX session...")

            try {
                // Load from file path (handles external data automatically if in same dir)
                ortSession = ortEnvironment.createSession(
                    modelFile.absolutePath,
                    sessionOptions
                )
                Log.d(TAG, "✓ Session created from file path")

            } catch (e: Exception) {
                Log.e(TAG, "Failed to load from file path, trying byte array fallback", e)

                // Fallback: Try loading as byte array (only works for single-file models)
                try {
                    val modelBytes = context.assets.open(Config.ONNX_MODEL_FILENAME).use {
                        it.readBytes()
                    }
                    ortSession = ortEnvironment.createSession(modelBytes, sessionOptions)
                    Log.d(TAG, "✓ Session created from byte array")
                } catch (e2: Exception) {
                    Log.e(TAG, "Both loading strategies failed", e2)
                    throw e2
                }
            }

            // Get input/output names
            inputName = ortSession?.inputNames?.iterator()?.next()
            outputName = ortSession?.outputNames?.iterator()?.next()

            Log.d(TAG, "✓ ONNX model loaded successfully")
            Log.d(TAG, "  Input: $inputName")
            Log.d(TAG, "  Output: $outputName")

            // Log model info
            ortSession?.inputInfo?.get(inputName)?.let { inputInfo ->
                Log.d(TAG, "  Input info: ${inputInfo.info}")
            }

        } catch (e: Exception) {
            val errorMsg = "Failed to load ONNX model: Error code - ${e.javaClass.simpleName} - message: ${e.message}"
            Log.e(TAG, errorMsg, e)

            // Log full stack trace for debugging
            e.printStackTrace()

            throw RuntimeException(errorMsg, e)
        }
    }

    /**
     * Copy asset file to internal storage (better for external data models)
     */
    private fun copyAssetToFile(context: Context, assetName: String): java.io.File {
        // Use files dir instead of cache (more reliable for external data)
        val filesDir = java.io.File(context.filesDir, "models")
        if (!filesDir.exists()) {
            filesDir.mkdirs()
            Log.d(TAG, "Created models directory: ${filesDir.absolutePath}")
        }

        val targetFile = java.io.File(filesDir, assetName)

        // Delete if exists to ensure fresh copy
        if (targetFile.exists()) {
            targetFile.delete()
            Log.d(TAG, "Deleted existing file: $assetName")
        }

        // Copy from assets
        try {
            context.assets.open(assetName).use { inputStream ->
                targetFile.outputStream().use { outputStream ->
                    val bytesWritten = inputStream.copyTo(outputStream)
                    outputStream.flush()
                    Log.d(TAG, "Copied $assetName: $bytesWritten bytes")
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to copy asset: $assetName", e)
            throw RuntimeException("Failed to copy asset $assetName: ${e.message}", e)
        }

        // Verify file was created
        if (!targetFile.exists() || targetFile.length() == 0L) {
            throw RuntimeException("Failed to create valid file: $assetName")
        }

        return targetFile
    }

    /**
     * Run inference on sequence
     *
     * @param sequence Shape: [sequence_length, num_features]
     * @return Pair of (predicted_class_index, confidence_scores)
     */
    fun predict(sequence: Array<FloatArray>): Pair<Int, FloatArray>? {
        val session = ortSession ?: run {
            Log.e(TAG, "ONNX session not initialized")
            return null
        }

        val input = inputName ?: run {
            Log.e(TAG, "Input name not found")
            return null
        }

        try {
            // Prepare input tensor
            // Expected shape: [batch=1, sequence_length=15, features=63]
            val batchSize = 1L
            val sequenceLength = Config.SEQUENCE_LENGTH.toLong()
            val numFeatures = Config.NUM_FEATURES.toLong()

            val shape = longArrayOf(batchSize, sequenceLength, numFeatures)

            // Flatten sequence to 1D array
            val flatSequence = FloatArray(Config.SEQUENCE_LENGTH * Config.NUM_FEATURES)
            var idx = 0
            for (frame in sequence) {
                for (value in frame) {
                    flatSequence[idx++] = value
                }
            }

            // Create tensor
            val inputTensor = OnnxTensor.createTensor(
                ortEnvironment,
                FloatBuffer.wrap(flatSequence),
                shape
            )

            // Run inference
            val results = session.run(mapOf(input to inputTensor))

            // Get output tensor (logits)
            val outputTensor = results[0].value as Array<*>
            val logits = (outputTensor[0] as FloatArray)

            // DEBUG: Log raw logits (first time only)
            if (logits.size == Config.NUM_CLASSES) {
                Log.d(TAG, "=== ONNX DEBUG (first 3 logits) ===")
                Log.d(TAG, "Raw logits: [${logits[0]}, ${logits[1]}, ${logits[2]}, ...]")
            }

            // Apply softmax to convert logits to probabilities
            val probabilities = applySoftmax(logits)

            // DEBUG: Log probabilities
            Log.d(TAG, "Probabilities: ${probabilities.take(5).joinToString { "%.2f%%".format(it * 100) }}")

            // Find predicted class (argmax)
            var maxIdx = 0
            var maxProb = probabilities[0]

            for (i in 1 until probabilities.size) {
                if (probabilities[i] > maxProb) {
                    maxProb = probabilities[i]
                    maxIdx = i
                }
            }

            Log.d(TAG, "Prediction: ${Config.IDX_TO_LABEL[maxIdx]} = %.1f%%".format(maxProb * 100))

            // Clean up
            inputTensor.close()
            results.close()

            return Pair(maxIdx, probabilities)

        } catch (e: Exception) {
            Log.e(TAG, "Inference failed", e)
            return null
        }
    }

    /**
     * Run inference with confidence threshold
     *
     * @param sequence Shape: [sequence_length, num_features]
     * @param threshold Minimum confidence threshold
     * @return Triple of (gesture_name, confidence, all_probabilities) or null
     */
    fun predictWithConfidence(
        sequence: Array<FloatArray>,
        threshold: Float = Config.CONFIDENCE_THRESHOLD
    ): Triple<String, Float, FloatArray>? {

        val (predictedIdx, probabilities) = predict(sequence) ?: return null

        val confidence = probabilities[predictedIdx]

        // Apply confidence threshold
        if (confidence < threshold) {
            return null
        }

        val gestureName = Config.IDX_TO_LABEL[predictedIdx] ?: "unknown"

        return Triple(gestureName, confidence, probabilities)
    }

    /**
     * Apply softmax to convert logits to probabilities
     */
    private fun applySoftmax(logits: FloatArray): FloatArray {
        // Find max for numerical stability
        val maxLogit = logits.maxOrNull() ?: 0f

        // Compute exp(logit - max)
        val expValues = FloatArray(logits.size)
        var sumExp = 0f

        for (i in logits.indices) {
            expValues[i] = kotlin.math.exp(logits[i] - maxLogit)
            sumExp += expValues[i]
        }

        // Normalize
        val probabilities = FloatArray(logits.size)
        for (i in logits.indices) {
            probabilities[i] = expValues[i] / sumExp
        }

        return probabilities
    }

    /**
     * Get model input shape
     */
    fun getInputShape(): String? {
        return try {
            ortSession?.inputInfo?.get(inputName)?.info?.toString()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Get model output shape
     */
    fun getOutputShape(): String? {
        return try {
            ortSession?.outputInfo?.get(outputName)?.info?.toString()
        } catch (e: Exception) {
            null
        }
    }

    /**
     * Release resources
     */
    fun close() {
        try {
            ortSession?.close()
            Log.d(TAG, "ONNX session closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing ONNX session", e)
        }
    }
}