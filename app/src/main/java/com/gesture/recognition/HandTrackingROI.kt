package com.gesture.recognition

import kotlin.math.*

/**
 * Hand Tracking ROI Generator
 *
 * Builds ROI from previous landmarks for tracking mode
 * Avoids running expensive detector when hand is already tracked
 *
 * Based on MediaPipe hand_landmarks_to_rect algorithm
 */
object HandTrackingROI {

    private const val TAG = "HandTrackingROI"

    // Same transformation constants as detector
    private const val SCALE_X = 2.9f
    private const val SCALE_Y = 2.9f
    private const val SHIFT_X = 0.0f
    private const val SHIFT_Y = -0.5f

    /**
     * Build tracking ROI from landmarks
     *
     * Uses landmarks to calculate bounding box and rotation
     * Much faster than running detector!
     *
     * @param landmarks Previous frame landmarks (21 × 3) in frame coordinates
     * @param frameWidth Frame width
     * @param frameHeight Frame height
     * @return HandROI for next frame
     */
    fun buildTrackingROI(
        landmarks: Array<FloatArray>,
        frameWidth: Int,
        frameHeight: Int
    ): HandROI {

        // Find bounding box of all landmarks
        var xMin = Float.MAX_VALUE
        var yMin = Float.MAX_VALUE
        var xMax = Float.MIN_VALUE
        var yMax = Float.MIN_VALUE

        for (lm in landmarks) {
            val x = lm[0]
            val y = lm[1]
            if (x < xMin) xMin = x
            if (x > xMax) xMax = x
            if (y < yMin) yMin = y
            if (y > yMax) yMax = y
        }

        // Box dimensions (in pixels)
        val boxW = xMax - xMin
        val boxH = yMax - yMin
        val boxCx = xMin + boxW / 2
        val boxCy = yMin + boxH / 2

        // Calculate rotation from wrist → index finger MCP
        // landmarks[0] = wrist
        // landmarks[5] = index finger MCP
        val wrist = landmarks[0]
        val indexMcp = landmarks[5]

        val dx = indexMcp[0] - wrist[0]
        val dy = indexMcp[1] - wrist[1]

        // MediaPipe rotation formula
        val rotation = normalizeRadians(
            PI.toFloat() / 2 - atan2(-dy, dx)
        )

        // Normalize box to [0, 1]
        val boxNormW = boxW / frameWidth
        val boxNormH = boxH / frameHeight
        val boxNormCx = boxCx / frameWidth
        val boxNormCy = boxCy / frameHeight

        // Apply transformation (same as detection ROI)
        val (cxA, cyA) = if (abs(rotation) < 1e-6) {
            Pair(
                (boxNormCx + boxNormW * SHIFT_X) * frameWidth,
                (boxNormCy + boxNormH * SHIFT_Y) * frameHeight
            )
        } else {
            val xShift = frameWidth * boxNormW * SHIFT_X * cos(rotation) -
                        frameHeight * boxNormH * SHIFT_Y * sin(rotation)
            val yShift = frameWidth * boxNormW * SHIFT_X * sin(rotation) +
                        frameHeight * boxNormH * SHIFT_Y * cos(rotation)
            Pair(
                boxNormCx * frameWidth + xShift,
                boxNormCy * frameHeight + yShift
            )
        }

        // Square ROI (scale by longer side)
        val longSide = maxOf(boxNormW * frameWidth, boxNormH * frameHeight)
        val wA = longSide * SCALE_X
        val hA = longSide * SCALE_Y

        // Calculate 4 corner points
        val rectPoints = rotatedRectToPoints(cxA, cyA, wA, hA, rotation)

        return HandROI(
            rotation = rotation,
            centerX = cxA,
            centerY = cyA,
            width = wA,
            height = hA,
            rectPoints = rectPoints,
            frameWidth = frameWidth,   // ← ADD
            frameHeight = frameHeight  // ← ADD
        )
    }

    /**
     * Calculate 4 corners of rotated rectangle
     */
    private fun rotatedRectToPoints(
        cx: Float, cy: Float,
        w: Float, h: Float,
        rotation: Float
    ): Array<FloatArray> {
        val b = cos(rotation) * 0.5f
        val a = sin(rotation) * 0.5f

        val p0x = cx - a * h - b * w
        val p0y = cy + b * h - a * w
        val p1x = cx + a * h - b * w
        val p1y = cy - b * h - a * w
        val p2x = 2 * cx - p0x
        val p2y = 2 * cy - p0y
        val p3x = 2 * cx - p1x
        val p3y = 2 * cx - p1y

        return arrayOf(
            floatArrayOf(p0x, p0y),  // bottom-left
            floatArrayOf(p1x, p1y),  // top-left
            floatArrayOf(p2x, p2y),  // top-right
            floatArrayOf(p3x, p3y)   // bottom-right
        )
    }

    /**
     * Normalize angle to [-π, π]
     */
    private fun normalizeRadians(angle: Float): Float {
        val twoPi = 2 * PI.toFloat()
        return angle - twoPi * floor((angle + PI.toFloat()) / twoPi)
    }
}