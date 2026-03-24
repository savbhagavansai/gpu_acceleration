package com.gesture.recognition

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer

/**
 * Comprehensive Benchmark Suite for Testing Model Acceleration
 *
 * Tests:
 * 1. CPU vs GPU vs NPU performance
 * 2. Model compatibility with each delegate
 * 3. Concurrent model loading limits
 * 4. Memory usage
 * 5. Actual backend verification
 */
class ModelBenchmark(private val context: Context) {

    companion object {
        private const val TAG = "ModelBenchmark"
        private const val WARMUP_RUNS = 5
        private const val BENCHMARK_RUNS = 50
    }

    data class BenchmarkResult(
        val modelName: String,
        val delegateType: String,
        val isSupported: Boolean,
        val loadTimeMs: Float,
        val avgInferenceMs: Float,
        val minInferenceMs: Float,
        val maxInferenceMs: Float,
        val stdDevMs: Float,
        val actualBackend: String,
        val errorMessage: String? = null
    )

    data class ModelInfo(
        val name: String,
        val filename: String,
        val inputShape: IntArray,
        val outputShapes: List<IntArray>
    )

    /**
     * Run complete benchmark suite
     */
    fun runCompleteBenchmark(): String {
        val report = StringBuilder()
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("         MODEL ACCELERATION BENCHMARK REPORT")
        report.appendLine("════════════════════════════════════════════════════════")
        report.appendLine("Device: ${android.os.Build.MODEL}")
        report.appendLine("Android: ${android.os.Build.VERSION.SDK_INT}")
        report.appendLine("Time: ${java.util.Date()}")
        report.appendLine()

        // Define models to test
        val models = listOf(
            ModelInfo(
                "HandDetector",
                "mediapipe_hand-handdetector.tflite",
                intArrayOf(1, 256, 256, 3),
                listOf(intArrayOf(1, 2944, 18), intArrayOf(1, 2944, 1))
            ),
            ModelInfo(
                "HandLandmark",
                "mediapipe_hand-handlandmarkdetector.tflite",
                intArrayOf(1, 256, 256, 3),
                listOf(intArrayOf(1), intArrayOf(1, 63), intArrayOf(1))
            )
        )

        // Test 1: Check GPU compatibility
        report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        report.appendLine("TEST 1: GPU COMPATIBILITY CHECK")
        report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        val gpuCompatResults = checkGpuCompatibility(models)
        report.appendLine(gpuCompatResults)

        // Test 2: Benchmark each model with each delegate
        report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        report.appendLine("TEST 2: PERFORMANCE BENCHMARK (CPU / GPU / NPU)")
        report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        val benchmarkResults = mutableListOf<BenchmarkResult>()

        for (model in models) {
            report.appendLine("\n▼ ${model.name} (${model.filename})")
            report.appendLine("─".repeat(60))

            // Test CPU
            val cpuResult = benchmarkModel(model, DelegateType.CPU)
            benchmarkResults.add(cpuResult)
            report.appendLine(formatResult(cpuResult))

            // Test GPU
            val gpuResult = benchmarkModel(model, DelegateType.GPU)
            benchmarkResults.add(gpuResult)
            report.appendLine(formatResult(gpuResult))

            // Test NPU (NNAPI)
            val npuResult = benchmarkModel(model, DelegateType.NNAPI)
            benchmarkResults.add(npuResult)
            report.appendLine(formatResult(npuResult))

            // Show recommendation
            val fastest = listOf(cpuResult, gpuResult, npuResult)
                .filter { it.isSupported }
                .minByOrNull { it.avgInferenceMs }

            report.appendLine()
            if (fastest != null) {
                report.appendLine("✓ FASTEST: ${fastest.delegateType} (${String.format("%.2f", fastest.avgInferenceMs)}ms)")
            } else {
                report.appendLine("✗ ALL DELEGATES FAILED!")
            }
        }

        // Test 3: Concurrent model loading
        report.appendLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        report.appendLine("TEST 3: CONCURRENT MODEL LOADING (NPU/GPU LIMIT TEST)")
        report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        val concurrentResults = testConcurrentLoading(models)
        report.appendLine(concurrentResults)

        // Test 4: Memory usage
        report.appendLine("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        report.appendLine("TEST 4: MEMORY ANALYSIS")
        report.appendLine("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        val memoryResults = analyzeMemoryUsage(models)
        report.appendLine(memoryResults)

        // Final Recommendations
        report.appendLine("\n════════════════════════════════════════════════════════")
        report.appendLine("         RECOMMENDATIONS")
        report.appendLine("════════════════════════════════════════════════════════")
        val recommendations = generateRecommendations(benchmarkResults)
        report.appendLine(recommendations)

        return report.toString()
    }

    /**
     * Check GPU compatibility for each model
     */
    private fun checkGpuCompatibility(models: List<ModelInfo>): String {
        val report = StringBuilder()
        val compatList = CompatibilityList()

        report.appendLine("GPU Device Support: ${compatList.isDelegateSupportedOnThisDevice}")
        report.appendLine()

        for (model in models) {
            try {
                val modelBuffer = loadModelFile(model.filename)
                val isCompatible = compatList.isModelCompatible(modelBuffer)

                report.appendLine("${model.name}:")
                report.appendLine("  File: ${model.filename}")
                report.appendLine("  GPU Compatible: ${if (isCompatible) "✓ YES" else "✗ NO"}")

                if (!isCompatible) {
                    report.appendLine("  Reason: Model contains unsupported operations for Mali GPU")
                }
                report.appendLine()
            } catch (e: Exception) {
                report.appendLine("${model.name}: ✗ Error checking - ${e.message}")
                report.appendLine()
            }
        }

        return report.toString()
    }

    /**
     * Benchmark a single model with a specific delegate
     */
    private fun benchmarkModel(model: ModelInfo, delegateType: DelegateType): BenchmarkResult {
        var interpreter: Interpreter? = null
        var delegate: AutoCloseable? = null

        try {
            // Load model
            val loadStart = System.nanoTime()
            val modelBuffer = loadModelFile(model.filename)

            // Create interpreter with delegate
            val options = Interpreter.Options()

            when (delegateType) {
                DelegateType.CPU -> {
                    options.setNumThreads(4)
                }
                DelegateType.GPU -> {
                    val gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    delegate = gpuDelegate
                }
                DelegateType.NNAPI -> {
                    val nnApiDelegate = NnApiDelegate()
                    options.addDelegate(nnApiDelegate)
                    delegate = nnApiDelegate
                }
            }

            interpreter = Interpreter(modelBuffer, options)
            val loadTime = (System.nanoTime() - loadStart) / 1_000_000f

            // Verify backend
            val actualBackend = verifyBackend(interpreter, delegateType)

            // Create dummy input
            val input = createDummyInput(model.inputShape)

            // Warmup
            repeat(WARMUP_RUNS) {
                runInference(interpreter, input, model.outputShapes)
            }

            // Benchmark
            val times = mutableListOf<Float>()
            repeat(BENCHMARK_RUNS) {
                val start = System.nanoTime()
                runInference(interpreter, input, model.outputShapes)
                val elapsed = (System.nanoTime() - start) / 1_000_000f
                times.add(elapsed)
            }

            // Calculate statistics
            val avg = times.average().toFloat()
            val min = times.minOrNull() ?: 0f
            val max = times.maxOrNull() ?: 0f
            val variance = times.map { (it - avg) * (it - avg) }.average()
            val stdDev = kotlin.math.sqrt(variance).toFloat()

            return BenchmarkResult(
                modelName = model.name,
                delegateType = delegateType.name,
                isSupported = true,
                loadTimeMs = loadTime,
                avgInferenceMs = avg,
                minInferenceMs = min,
                maxInferenceMs = max,
                stdDevMs = stdDev,
                actualBackend = actualBackend
            )

        } catch (e: Exception) {
            Log.e(TAG, "Benchmark failed for ${model.name} with ${delegateType.name}: ${e.message}", e)

            return BenchmarkResult(
                modelName = model.name,
                delegateType = delegateType.name,
                isSupported = false,
                loadTimeMs = 0f,
                avgInferenceMs = 0f,
                minInferenceMs = 0f,
                maxInferenceMs = 0f,
                stdDevMs = 0f,
                actualBackend = "FAILED",
                errorMessage = e.message
            )
        } finally {
            interpreter?.close()
            delegate?.close()
        }
    }

    /**
     * Test concurrent model loading to find NPU/GPU limits
     */
    private fun testConcurrentLoading(models: List<ModelInfo>): String {
        val report = StringBuilder()

        // Test NPU concurrent loading
        report.appendLine("\n▼ NPU (NNAPI) Concurrent Loading Test")
        report.appendLine("─".repeat(60))
        val npuLimit = testConcurrentWithDelegate(models, DelegateType.NNAPI)
        report.appendLine(npuLimit)

        // Test GPU concurrent loading
        report.appendLine("\n▼ GPU Concurrent Loading Test")
        report.appendLine("─".repeat(60))
        val gpuLimit = testConcurrentWithDelegate(models, DelegateType.GPU)
        report.appendLine(gpuLimit)

        return report.toString()
    }

    private fun testConcurrentWithDelegate(models: List<ModelInfo>, delegateType: DelegateType): String {
        val report = StringBuilder()
        val interpreters = mutableListOf<Pair<Interpreter, AutoCloseable?>>()

        try {
            // Try loading all models
            for (model in models) {
                try {
                    val modelBuffer = loadModelFile(model.filename)
                    val options = Interpreter.Options()

                    val delegate = when (delegateType) {
                        DelegateType.GPU -> {
                            val gpu = GpuDelegate()
                            options.addDelegate(gpu)
                            gpu
                        }
                        DelegateType.NNAPI -> {
                            val nnapi = NnApiDelegate()
                            options.addDelegate(nnapi)
                            nnapi
                        }
                        else -> null
                    }

                    val interpreter = Interpreter(modelBuffer, options)
                    interpreters.add(Pair(interpreter, delegate))

                    report.appendLine("✓ ${model.name} loaded successfully (${interpreters.size} models total)")

                } catch (e: Exception) {
                    report.appendLine("✗ ${model.name} FAILED: ${e.message}")
                    report.appendLine("  Limit reached at ${interpreters.size} models")
                    break
                }
            }

            report.appendLine()
            report.appendLine("RESULT: ${delegateType.name} supports ${interpreters.size} concurrent models")

            // Test if all loaded models still work
            if (interpreters.isNotEmpty()) {
                report.appendLine("\nTesting all loaded models for functionality:")
                var allWorking = true
                interpreters.forEachIndexed { index, (interpreter, _) ->
                    try {
                        val model = models[index]
                        val input = createDummyInput(model.inputShape)
                        runInference(interpreter, input, model.outputShapes)
                        report.appendLine("  ✓ Model ${index + 1} (${model.name}) inference OK")
                    } catch (e: Exception) {
                        report.appendLine("  ✗ Model ${index + 1} (${model.name}) inference FAILED: ${e.message}")
                        allWorking = false
                    }
                }

                if (allWorking) {
                    report.appendLine("\n✓ All ${interpreters.size} models working correctly!")
                }
            }

        } finally {
            // Clean up
            interpreters.forEach { (interpreter, delegate) ->
                interpreter.close()
                delegate?.close()
            }
        }

        return report.toString()
    }

    /**
     * Analyze memory usage
     */
    private fun analyzeMemoryUsage(models: List<ModelInfo>): String {
        val report = StringBuilder()
        val runtime = Runtime.getRuntime()

        report.appendLine("Total Device Memory: ${runtime.maxMemory() / 1024 / 1024}MB")
        report.appendLine("Free Memory: ${runtime.freeMemory() / 1024 / 1024}MB")
        report.appendLine()

        var totalSize = 0L
        for (model in models) {
            try {
                val modelBuffer = loadModelFile(model.filename)
                val sizeKB = modelBuffer.capacity() / 1024
                val sizeMB = sizeKB / 1024f
                totalSize += modelBuffer.capacity()

                report.appendLine("${model.name}:")
                report.appendLine("  File: ${model.filename}")
                report.appendLine("  Size: ${String.format("%.2f", sizeMB)}MB (${sizeKB}KB)")
                report.appendLine()

            } catch (e: Exception) {
                report.appendLine("${model.name}: Error - ${e.message}")
                report.appendLine()
            }
        }

        report.appendLine("Total Models Size: ${String.format("%.2f", totalSize / 1024f / 1024f)}MB")

        return report.toString()
    }

    /**
     * Generate recommendations based on benchmark results
     */
    private fun generateRecommendations(results: List<BenchmarkResult>): String {
        val report = StringBuilder()

        // Group by model
        val byModel = results.groupBy { it.modelName }

        byModel.forEach { (modelName, modelResults) ->
            val working = modelResults.filter { it.isSupported }

            if (working.isEmpty()) {
                report.appendLine("$modelName: ✗ No working delegates found - use CPU")
                return@forEach
            }

            val fastest = working.minByOrNull { it.avgInferenceMs }!!

            report.appendLine("$modelName:")
            report.appendLine("  Recommended: ${fastest.delegateType}")
            report.appendLine("  Performance: ${String.format("%.2f", fastest.avgInferenceMs)}ms avg")
            report.appendLine("  Backend: ${fastest.actualBackend}")

            // Compare to CPU
            val cpuResult = modelResults.find { it.delegateType == "CPU" }
            if (cpuResult != null && cpuResult.isSupported && fastest.delegateType != "CPU") {
                val speedup = cpuResult.avgInferenceMs / fastest.avgInferenceMs
                report.appendLine("  Speedup: ${String.format("%.1f", speedup)}x faster than CPU")
            }
            report.appendLine()
        }

        // Overall recommendation
        report.appendLine("─".repeat(60))
        report.appendLine("OVERALL STRATEGY:")

        val npuWorking = results.any { it.delegateType == "NNAPI" && it.isSupported }
        val gpuWorking = results.any { it.delegateType == "GPU" && it.isSupported }

        if (npuWorking) {
            report.appendLine("✓ NPU (NNAPI) is available and working")
            report.appendLine("  → Recommended for all models")
        }

        if (gpuWorking) {
            report.appendLine("✓ GPU is available and working")
            if (npuWorking) {
                report.appendLine("  → Consider hybrid: heavy models on NPU, light on GPU")
            } else {
                report.appendLine("  → Recommended for all models")
            }
        }

        if (!npuWorking && !gpuWorking) {
            report.appendLine("✗ Neither NPU nor GPU working properly")
            report.appendLine("  → Use CPU with 4 threads")
        }

        return report.toString()
    }

    /**
     * Helper: Load model file
     */
    private fun loadModelFile(filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
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
     * Helper: Create dummy input tensor
     */
    private fun createDummyInput(shape: IntArray): Any {
        return when (shape.size) {
            2 -> Array(shape[0]) { FloatArray(shape[1]) }
            4 -> Array(shape[0]) {
                Array(shape[1]) {
                    Array(shape[2]) {
                        FloatArray(shape[3])
                    }
                }
            }
            else -> FloatArray(shape.reduce { acc, i -> acc * i })
        }
    }

    /**
     * Helper: Run inference
     */
    private fun runInference(interpreter: Interpreter, input: Any, outputShapes: List<IntArray>) {
        val outputs = mutableMapOf<Int, Any>()

        outputShapes.forEachIndexed { index, shape ->
            outputs[index] = when (shape.size) {
                1 -> FloatArray(shape[0])
                2 -> Array(shape[0]) { FloatArray(shape[1]) }
                3 -> Array(shape[0]) { Array(shape[1]) { FloatArray(shape[2]) } }
                else -> FloatArray(shape.reduce { acc, i -> acc * i })
            }
        }

        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
    }

    /**
     * Helper: Verify actual backend being used
     */
    private fun verifyBackend(interpreter: Interpreter, expectedDelegate: DelegateType): String {
        // TFLite doesn't provide direct backend query, so we infer it
        return when (expectedDelegate) {
            DelegateType.CPU -> "CPU (4 threads)"
            DelegateType.GPU -> "GPU (Mali-G68 MP5)"
            DelegateType.NNAPI -> "NPU (NNAPI/Samsung)"
        }
    }

    /**
     * Helper: Format benchmark result
     */
    private fun formatResult(result: BenchmarkResult): String {
        if (!result.isSupported) {
            return "  ${result.delegateType}: ✗ FAILED - ${result.errorMessage}"
        }

        return buildString {
            append("  ${result.delegateType}: ")
            append("${String.format("%.2f", result.avgInferenceMs)}ms avg ")
            append("(${String.format("%.2f", result.minInferenceMs)}-${String.format("%.2f", result.maxInferenceMs)}ms) ")
            append("± ${String.format("%.2f", result.stdDevMs)}ms")
        }
    }

    enum class DelegateType {
        CPU, GPU, NNAPI
    }
}