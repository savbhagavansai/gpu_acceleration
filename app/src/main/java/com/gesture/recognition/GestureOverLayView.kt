package com.gesture.recognition

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.View

/**
 * Custom overlay view with proper aspect ratio handling
 * - Hand skeleton (21 landmarks + connections)
 * - Top panel (status, buffer, gesture, confidence bar)
 * - Right panel (probability bars for all 11 classes)
 * - Correct scaling to match camera aspect ratio
 */
class GestureOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val TAG = "GestureOverlayView"

    // Data to display
    private var result: GestureResult? = null
    private var landmarks: FloatArray? = null

    // OPTIMIZATION: Use FloatArray instead of List<Pair> to avoid allocations
    // Stores x,y pairs: [x0, y0, x1, y1, ..., x20, y20] = 42 floats
    private var displayPoints: FloatArray? = null

    private var fps: Float = 0f
    private var frameCount: Int = 0
    private var bufferSize: Int = 0
    private var handDetected: Boolean = false
    private var imageWidth: Int = 320
    private var imageHeight: Int = 240
    private var rotation: Int = 0
    private var mirrorHorizontal: Boolean = false

    // OPTIMIZATION: Cache aspect ratio calculations (recalculate only on resize)
    private var cachedViewWidth = 0f
    private var cachedViewHeight = 0f
    private var cachedImageWidth = 0
    private var cachedImageHeight = 0
    private var cachedScaleX = 0f
    private var cachedScaleY = 0f
    private var cachedOffsetX = 0f
    private var cachedOffsetY = 0f
    private var cacheValid = false

    // PERFORMANCE MONITORING - Debug panel ALWAYS VISIBLE
    private var showDebugPanel = true  // ✅ CHANGED: true (always show)
    private var lastTapTime = 0L
    private var tapCount = 0

    // Timing data - UPDATED for ONNX pipeline
    private var handDetectorMs = 0.0
    private var landmarksMs = 0.0
    private var gestureMs = 0.0
    private var totalMs = 0.0
    private var wasTracking = false

    // Expected performance targets - UPDATED for ONNX pipeline
    private val expectedHandDetectorMs = 3.0   // NPU target for detector
    private val expectedLandmarksMs = 2.0      // NPU target for landmarks
    private val expectedGestureMs = 3.0        // NPU target for gesture
    private val expectedTotalMs = 10.0         // Total target

    // Thread safety - create copy before drawing
    private val landmarksLock = Any()

    // MediaPipe hand connections (21 landmarks)
    private val handConnections = listOf(
        // Thumb
        0 to 1, 1 to 2, 2 to 3, 3 to 4,
        // Index
        0 to 5, 5 to 6, 6 to 7, 7 to 8,
        // Middle
        0 to 9, 9 to 10, 10 to 11, 11 to 12,
        // Ring
        0 to 13, 13 to 14, 14 to 15, 15 to 16,
        // Pinky
        0 to 17, 17 to 18, 18 to 19, 19 to 20,
        // Palm
        5 to 9, 9 to 13, 13 to 17
    )

    // Paints - UPDATED: Larger and more visible
    private val landmarkPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val connectionPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 8f  // ✅ CHANGED: 4f → 8f (thicker lines)
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 40f
        isAntiAlias = true
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private val smallTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 30f
        isAntiAlias = true
    }

    private val tinyTextPaint = Paint().apply {
        color = Color.WHITE
        textSize = 24f
        isAntiAlias = true
    }

    private val backgroundPaint = Paint().apply {
        color = Color.argb(180, 0, 0, 0)
        style = Paint.Style.FILL
    }

    private val barBackgroundPaint = Paint().apply {
        color = Color.argb(100, 100, 100, 100)
        style = Paint.Style.FILL
    }

    /**
     * Update all data at once (thread-safe)
     * Pre-computes display coordinates for faster rendering
     */
    fun updateData(
        result: GestureResult?,
        landmarks: FloatArray?,
        fps: Float,
        frameCount: Int,
        bufferSize: Int,
        handDetected: Boolean,
        imageWidth: Int,
        imageHeight: Int,
        rotation: Int,
        mirrorHorizontal: Boolean
    ) {
        synchronized(landmarksLock) {
            this.result = result
            this.landmarks = landmarks?.copyOf()
            this.fps = fps
            this.frameCount = frameCount
            this.bufferSize = bufferSize
            this.handDetected = handDetected
            this.imageWidth = imageWidth
            this.imageHeight = imageHeight
            this.rotation = rotation
            this.mirrorHorizontal = mirrorHorizontal

            // Extract timing data from result for performance monitoring - UPDATED
            this.handDetectorMs = result?.handDetectorTimeMs ?: 0.0
            this.landmarksMs = result?.landmarksTimeMs ?: 0.0
            this.gestureMs = result?.gestureTimeMs ?: 0.0
            this.totalMs = result?.totalTimeMs ?: 0.0
            this.wasTracking = result?.wasTracking ?: false

            // PRE-COMPUTE display coordinates (do heavy math here, not in onDraw!)
            this.displayPoints = if (landmarks != null && landmarks.size == 63) {
                preComputeDisplayPoints(landmarks, imageWidth, imageHeight, rotation, mirrorHorizontal)
            } else {
                null
            }
        }
        postInvalidate()
    }

    /**
     * Pre-compute all display coordinates (runs in background, not on UI thread!)
     * OPTIMIZED: Uses cached scale/offset values and FloatArray instead of List<Pair>
     */
    private fun preComputeDisplayPoints(
        lm: FloatArray,
        imageWidth: Int,
        imageHeight: Int,
        rotation: Int,
        mirror: Boolean
    ): FloatArray {
        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        if (viewWidth == 0f || viewHeight == 0f) {
            return FloatArray(0)
        }

        // Check if cache is invalid (view or image dimensions changed)
        if (!cacheValid ||
            viewWidth != cachedViewWidth ||
            viewHeight != cachedViewHeight ||
            imageWidth != cachedImageWidth ||
            imageHeight != cachedImageHeight) {

            // Recalculate and cache aspect ratio scaling
            val imageAspect = imageWidth.toFloat() / imageHeight
            val viewAspect = viewWidth / viewHeight

            if (imageAspect > viewAspect) {
                cachedScaleX = viewWidth
                cachedScaleY = viewWidth / imageAspect
                cachedOffsetX = 0f
                cachedOffsetY = (viewHeight - cachedScaleY) / 2f
            } else {
                cachedScaleX = viewHeight * imageAspect
                cachedScaleY = viewHeight
                cachedOffsetX = (viewWidth - cachedScaleX) / 2f
                cachedOffsetY = 0f
            }

            // Update cache
            cachedViewWidth = viewWidth
            cachedViewHeight = viewHeight
            cachedImageWidth = imageWidth
            cachedImageHeight = imageHeight
            cacheValid = true
        }

        // OPTIMIZATION: Use FloatArray instead of List<Pair>
        // 21 landmarks × 2 coordinates = 42 floats
        val points = FloatArray(42)

        // Transform all 21 landmarks using CACHED scale/offset
        for (i in 0 until 21) {
            // FIXED: Landmarks are in PIXEL coordinates, normalize them first
            val rawX = lm[i * 3] / imageWidth.toFloat()  // Normalize to [0,1]
            val rawY = lm[i * 3 + 1] / imageHeight.toFloat()  // Normalize to [0,1]

            // Apply rotation
            val (rotatedX, rotatedY) = transformCoordinate(rawX, rawY, rotation)

            // Apply mirroring
            val finalX = if (mirror) 1.0f - rotatedX else rotatedX
            val finalY = rotatedY

            // Scale to view coordinates using CACHED values
            points[i * 2] = finalX * cachedScaleX + cachedOffsetX
            points[i * 2 + 1] = finalY * cachedScaleY + cachedOffsetY
        }

        return points
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        try {
            // Draw in order: back to front
            drawHandSkeleton(canvas)
            drawTopPanel(canvas)
            drawDebugPanel(canvas)  // ← Performance monitoring panel
            drawProbabilityPanel(canvas)
            drawBottomInstructions(canvas)
        } catch (e: Exception) {
            Log.e(TAG, "Error drawing overlay", e)
        }
    }

    /**
     * Draw hand skeleton using PRE-COMPUTED display coordinates
     * (No transformation math here - already done in updateData!)
     * OPTIMIZED: Uses FloatArray for zero-allocation drawing
     */
    private fun drawHandSkeleton(canvas: Canvas) {
        // Use pre-computed display points (FloatArray: [x0,y0, x1,y1, ...])
        val points = synchronized(landmarksLock) {
            displayPoints
        } ?: run {
            Log.d(TAG, "No display points available")
            return
        }

        if (points.isEmpty() || points.size != 42) {  // 21 landmarks × 2 = 42
            Log.d(TAG, "Invalid points size: ${points.size}, expected 42")
            return
        }

        Log.d(TAG, "Drawing landmarks: ${points.size/2} points")

        // Draw connections first (underneath)
        for ((start, end) in handConnections) {
            if (start * 2 + 1 < points.size && end * 2 + 1 < points.size) {
                val x1 = points[start * 2]
                val y1 = points[start * 2 + 1]
                val x2 = points[end * 2]
                val y2 = points[end * 2 + 1]
                canvas.drawLine(x1, y1, x2, y2, connectionPaint)
            }
        }

        // Draw landmarks on top - LARGER circles for visibility
        for (i in 0 until 21) {
            val x = points[i * 2]
            val y = points[i * 2 + 1]
            canvas.drawCircle(x, y, 15f, landmarkPaint)  // ✅ CHANGED: 10f → 15f (larger)
        }
    }

    /**
     * Transform coordinate based on rotation
     */
    private fun transformCoordinate(x: Float, y: Float, rotation: Int): Pair<Float, Float> {
        return when (rotation) {
            90 -> {
                // 90° counter-clockwise
                Pair(1.0f - y, x)
            }
            180 -> {
                // 180° rotation
                Pair(1.0f - x, 1.0f - y)
            }
            270 -> {
                // 270° counter-clockwise (90° clockwise)
                Pair(y, 1.0f - x)
            }
            else -> {
                // No rotation
                Pair(x, y)
            }
        }
    }

    /**
     * Draw top panel with status, buffer, gesture, confidence
     */
    private fun drawTopPanel(canvas: Canvas) {
        val panelHeight = 250f

        // Semi-transparent background
        canvas.drawRect(20f, 20f, width - 20f, panelHeight, backgroundPaint)

        var y = 60f

        // Hand detection status
        if (handDetected) {
            textPaint.color = Color.GREEN
            canvas.drawText("✓ HAND DETECTED", 40f, y, textPaint)
        } else {
            textPaint.color = Color.RED
            canvas.drawText("✗ NO HAND", 40f, y, textPaint)
        }

        y += 50f

        // Buffer status
        val bufferText = "Buffer: $bufferSize/${Config.SEQUENCE_LENGTH}"
        if (bufferSize >= Config.SEQUENCE_LENGTH) {
            smallTextPaint.color = Color.GREEN
        } else {
            smallTextPaint.color = Color.YELLOW
        }
        canvas.drawText(bufferText, 40f, y, smallTextPaint)

        y += 50f

        // Gesture name
        val res = result
        if (res != null && res.meetsThreshold()) {
            val gestureName = res.getFormattedGesture().uppercase()
            textPaint.color = if (res.confidence > 0.8f) Color.GREEN
                             else Color.YELLOW
            canvas.drawText("GESTURE: $gestureName", 40f, y, textPaint)

            y += 50f

            // Confidence bar
            drawConfidenceBar(canvas, y, res.confidence)
        } else if (bufferSize < Config.SEQUENCE_LENGTH) {
            textPaint.color = Color.GRAY
            val progress = (bufferSize * 100) / Config.SEQUENCE_LENGTH
            canvas.drawText("Collecting frames... $progress%", 40f, y, textPaint)
        } else {
            textPaint.color = Color.GRAY
            canvas.drawText("Low confidence", 40f, y, textPaint)
        }

        // FPS (top right)
        textPaint.color = Color.WHITE
        val fpsText = "FPS: %.1f".format(fps)
        canvas.drawText(fpsText, width - 200f, 60f, textPaint)

        // Frame counter
        smallTextPaint.color = Color.WHITE
        canvas.drawText("Frame: $frameCount", width - 200f, 110f, smallTextPaint)
    }

    /**
     * Draw confidence bar
     */
    private fun drawConfidenceBar(canvas: Canvas, y: Float, confidence: Float) {
        val barX = 40f
        val barWidth = 500f
        val barHeight = 40f

        // Background
        canvas.drawRect(barX, y, barX + barWidth, y + barHeight, barBackgroundPaint)

        // Filled portion
        val fillWidth = barWidth * confidence
        val barColor = when {
            confidence > 0.8f -> Color.GREEN
            confidence > 0.6f -> Color.YELLOW
            else -> Color.rgb(255, 165, 0)  // Orange
        }

        val fillPaint = Paint().apply {
            color = barColor
            style = Paint.Style.FILL
        }
        canvas.drawRect(barX, y, barX + fillWidth, y + barHeight, fillPaint)

        // Border
        val borderPaint = Paint().apply {
            color = Color.WHITE
            style = Paint.Style.STROKE
            strokeWidth = 2f
        }
        canvas.drawRect(barX, y, barX + barWidth, y + barHeight, borderPaint)

        // Percentage text
        smallTextPaint.color = Color.WHITE
        canvas.drawText("${(confidence * 100).toInt()}%", barX + barWidth + 20f, y + 30f, smallTextPaint)
    }

    /**
     * Draw probability bars for all 11 classes (right side)
     */
    private fun drawProbabilityPanel(canvas: Canvas) {
        val res = result ?: return
        val probs = res.allProbabilities

        if (probs.isEmpty()) return

        val panelX = width - 350f
        val panelY = 300f
        val panelWidth = 330f
        val panelHeight = (Config.NUM_CLASSES * 45 + 70).toFloat()

        // Background
        canvas.drawRect(panelX, panelY, panelX + panelWidth, panelY + panelHeight, backgroundPaint)

        // Title
        textPaint.color = Color.WHITE
        canvas.drawText("Probabilities:", panelX + 20f, panelY + 40f, textPaint)

        // Draw bars
        var barY = panelY + 70f
        for (i in 0 until minOf(Config.NUM_CLASSES, probs.size)) {
            val label = Config.IDX_TO_LABEL[i] ?: "unknown"
            val prob = probs[i]

            // Label
            val labelShort = label.replace('_', ' ').take(12)
            tinyTextPaint.color = Color.WHITE
            canvas.drawText(labelShort, panelX + 20f, barY + 20f, tinyTextPaint)

            // Bar
            val barStartX = panelX + 150f
            val barMaxWidth = 120f
            val barActualWidth = barMaxWidth * prob
            val barPaint = Paint().apply {
                color = if (prob > 0.5f) Color.GREEN else Color.GRAY
                style = Paint.Style.FILL
            }
            canvas.drawRect(barStartX, barY, barStartX + barActualWidth, barY + 25f, barPaint)

            // Percentage
            tinyTextPaint.color = Color.WHITE
            canvas.drawText("${(prob * 100).toInt()}%", barStartX + barMaxWidth + 10f, barY + 20f, tinyTextPaint)

            barY += 45f
        }
    }

    /**
     * Draw performance debug panel (toggleable with triple-tap)
     * Shows HandDetector, Landmarks, Gesture timing and hardware status
     */
    private fun drawDebugPanel(canvas: Canvas) {
        if (!showDebugPanel) return

        val panelX = 40f
        val panelY = 180f
        val panelWidth = width - 80f
        val panelHeight = 550f  // Increased height for tracking mode

        // Semi-transparent background
        backgroundPaint.alpha = 230
        canvas.drawRoundRect(
            panelX, panelY,
            panelX + panelWidth, panelY + panelHeight,
            20f, 20f, backgroundPaint
        )

        // Title
        textPaint.color = Color.WHITE
        textPaint.textSize = 36f
        canvas.drawText("PERFORMANCE MONITOR", panelX + 20f, panelY + 50f, textPaint)

        smallTextPaint.color = Color.GRAY
        smallTextPaint.textSize = 24f
        canvas.drawText("Real-time Performance Metrics", panelX + 20f, panelY + 80f, smallTextPaint)  // ✅ CHANGED

        var yPos = panelY + 130f

        // HandDetector timing with color coding
        val detectorColor = if (handDetectorMs <= expectedHandDetectorMs * 1.5) Color.GREEN else Color.RED
        val detectorStatus = if (handDetectorMs <= expectedHandDetectorMs * 1.5) "✓ NPU" else "✗ CPU"

        smallTextPaint.color = Color.WHITE
        smallTextPaint.textSize = 28f
        canvas.drawText("HandDetector:", panelX + 20f, yPos, smallTextPaint)

        textPaint.color = detectorColor
        textPaint.textSize = 32f
        canvas.drawText(String.format("%.1fms", handDetectorMs), panelX + 250f, yPos, textPaint)

        smallTextPaint.color = detectorColor
        canvas.drawText(detectorStatus, panelX + 380f, yPos, smallTextPaint)

        yPos += 35f
        tinyTextPaint.color = Color.GRAY
        tinyTextPaint.textSize = 22f
        canvas.drawText(String.format("Target: %.0fms (NPU)", expectedHandDetectorMs), panelX + 40f, yPos, tinyTextPaint)

        yPos += 50f

        // Landmarks timing with color coding
        val landmarksColor = if (landmarksMs <= expectedLandmarksMs * 1.5) Color.GREEN else Color.RED
        val landmarksStatus = if (landmarksMs <= expectedLandmarksMs * 1.5) "✓ NPU" else "✗ CPU"

        smallTextPaint.color = Color.WHITE
        canvas.drawText("Landmarks:", panelX + 20f, yPos, smallTextPaint)

        textPaint.color = landmarksColor
        canvas.drawText(String.format("%.1fms", landmarksMs), panelX + 250f, yPos, textPaint)

        smallTextPaint.color = landmarksColor
        canvas.drawText(landmarksStatus, panelX + 380f, yPos, smallTextPaint)

        yPos += 35f
        tinyTextPaint.color = Color.GRAY
        canvas.drawText(String.format("Target: %.0fms (NPU)", expectedLandmarksMs), panelX + 40f, yPos, tinyTextPaint)

        yPos += 50f

        // Gesture timing with color coding
        val gestureColor = if (gestureMs <= expectedGestureMs * 1.5) Color.GREEN else Color.RED
        val gestureStatus = if (gestureMs <= expectedGestureMs * 1.5) "✓ NPU" else "✗ CPU"

        smallTextPaint.color = Color.WHITE
        canvas.drawText("Gesture:", panelX + 20f, yPos, smallTextPaint)

        textPaint.color = gestureColor
        canvas.drawText(String.format("%.1fms", gestureMs), panelX + 250f, yPos, textPaint)

        smallTextPaint.color = gestureColor
        canvas.drawText(gestureStatus, panelX + 380f, yPos, smallTextPaint)

        yPos += 35f
        tinyTextPaint.color = Color.GRAY
        canvas.drawText(String.format("Target: %.0fms (NPU)", expectedGestureMs), panelX + 40f, yPos, tinyTextPaint)

        yPos += 50f

        // Total timing
        val totalColor = if (totalMs <= expectedTotalMs * 1.5) Color.GREEN else Color.RED

        smallTextPaint.color = Color.WHITE
        canvas.drawText("Total:", panelX + 20f, yPos, smallTextPaint)

        textPaint.color = totalColor
        canvas.drawText(String.format("%.1fms", totalMs), panelX + 250f, yPos, textPaint)

        smallTextPaint.color = totalColor
        val totalStatus = if (totalMs <= expectedTotalMs * 1.5) "✓ FAST" else "✗ SLOW"
        canvas.drawText(totalStatus, panelX + 380f, yPos, smallTextPaint)

        yPos += 35f
        tinyTextPaint.color = Color.GRAY
        canvas.drawText(String.format("Target: %.0fms", expectedTotalMs), panelX + 40f, yPos, tinyTextPaint)

        yPos += 50f

        // Tracking Mode Indicator
        val modeColor = if (wasTracking) Color.CYAN else Color.YELLOW
        val modeText = if (wasTracking) "TRACKING" else "DETECTION"

        smallTextPaint.color = Color.WHITE
        canvas.drawText("Mode:", panelX + 20f, yPos, smallTextPaint)

        textPaint.color = modeColor
        textPaint.textSize = 32f
        canvas.drawText(modeText, panelX + 250f, yPos, textPaint)

        yPos += 45f

        // Frame info
        tinyTextPaint.color = Color.GRAY
        tinyTextPaint.textSize = 24f
        canvas.drawText("Frame: $frameCount | FPS: ${String.format("%.1f", fps)}",
                       panelX + 20f, yPos, tinyTextPaint)
    }

    /**
     * Draw bottom instructions
     */
    private fun drawBottomInstructions(canvas: Canvas) {
        val instructions = "Double tap to switch camera  •  Optimized for edge devices"
        tinyTextPaint.color = Color.WHITE
        canvas.drawText(instructions, 40f, height - 40f, tinyTextPaint)
    }

    /**
     * OPTIMIZATION: Invalidate cache when view size changes
     */
    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        cacheValid = false  // Force recalculation of scale/offset
        Log.d(TAG, "View resized: ${w}×${h}, cache invalidated")
    }

    /**
     * Handle touch events - REMOVED triple-tap (panel always visible now)
     */
    override fun onTouchEvent(event: MotionEvent): Boolean {
        // Panel is always visible, no toggle needed
        return false  // Let other views handle touch events
    }
}