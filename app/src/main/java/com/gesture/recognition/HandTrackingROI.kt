package com.gesture.recognition

/**
 * Region of Interest (ROI) for hand tracking
 *
 * Represents a rectangular region in the image where the hand is located.
 * Used for:
 * - Cropping the image to focus on the hand
 * - Tracking hand position between frames
 * - Affine transformation for landmark detection
 */
data class HandTrackingROI(
    val centerX: Float,      // Center X coordinate in pixels
    val centerY: Float,      // Center Y coordinate in pixels
    val roiWidth: Float,     // ROI width in pixels
    val roiHeight: Float,    // ROI height in pixels
    val rotation: Float      // Rotation angle in degrees (clockwise)
) {

    /**
     * Get bounding box coordinates [left, top, right, bottom]
     */
    fun getBoundingBox(): FloatArray {
        val left = centerX - roiWidth / 2
        val top = centerY - roiHeight / 2
        val right = centerX + roiWidth / 2
        val bottom = centerY + roiHeight / 2

        return floatArrayOf(left, top, right, bottom)
    }

    /**
     * Get corner points of the ROI (accounting for rotation)
     * Returns array of [x0, y0, x1, y1, x2, y2, x3, y3]
     * Order: top-left, top-right, bottom-right, bottom-left
     */
    fun getCornerPoints(): FloatArray {
        val halfWidth = roiWidth / 2
        val halfHeight = roiHeight / 2

        // Rotation in radians
        val radians = Math.toRadians(rotation.toDouble()).toFloat()
        val cos = Math.cos(radians.toDouble()).toFloat()
        val sin = Math.sin(radians.toDouble()).toFloat()

        // Corner offsets (before rotation)
        val corners = floatArrayOf(
            -halfWidth, -halfHeight,  // Top-left
            halfWidth, -halfHeight,   // Top-right
            halfWidth, halfHeight,    // Bottom-right
            -halfWidth, halfHeight    // Bottom-left
        )

        // Apply rotation and translate to center
        val rotatedCorners = FloatArray(8)
        for (i in 0 until 4) {
            val x = corners[i * 2]
            val y = corners[i * 2 + 1]

            rotatedCorners[i * 2] = centerX + (x * cos - y * sin)
            rotatedCorners[i * 2 + 1] = centerY + (x * sin + y * cos)
        }

        return rotatedCorners
    }

    /**
     * Check if ROI is valid (within image bounds)
     */
    fun isValid(imageWidth: Int, imageHeight: Int): Boolean {
        val box = getBoundingBox()
        return box[0] >= 0 && box[1] >= 0 &&
               box[2] <= imageWidth && box[3] <= imageHeight
    }

    /**
     * Clamp ROI to image bounds
     */
    fun clampToImage(imageWidth: Int, imageHeight: Int): HandTrackingROI {
        val box = getBoundingBox()

        val clampedLeft = box[0].coerceIn(0f, imageWidth.toFloat())
        val clampedTop = box[1].coerceIn(0f, imageHeight.toFloat())
        val clampedRight = box[2].coerceIn(0f, imageWidth.toFloat())
        val clampedBottom = box[3].coerceIn(0f, imageHeight.toFloat())

        val newCenterX = (clampedLeft + clampedRight) / 2
        val newCenterY = (clampedTop + clampedBottom) / 2
        val newWidth = clampedRight - clampedLeft
        val newHeight = clampedBottom - clampedTop

        return HandTrackingROI(
            centerX = newCenterX,
            centerY = newCenterY,
            roiWidth = newWidth,
            roiHeight = newHeight,
            rotation = rotation
        )
    }

    /**
     * Expand ROI by a scale factor
     * @param factor Scale factor (1.5 = 150% of original size)
     */
    fun expand(factor: Float): HandTrackingROI {
        return HandTrackingROI(
            centerX = centerX,
            centerY = centerY,
            roiWidth = roiWidth * factor,
            roiHeight = roiHeight * factor,
            rotation = rotation
        )
    }

    /**
     * Make ROI square by using the larger dimension
     */
    fun toSquare(): HandTrackingROI {
        val size = maxOf(roiWidth, roiHeight)
        return HandTrackingROI(
            centerX = centerX,
            centerY = centerY,
            roiWidth = size,
            roiHeight = size,
            rotation = rotation
        )
    }

    /**
     * Shift ROI by offset
     */
    fun shift(dx: Float, dy: Float): HandTrackingROI {
        return HandTrackingROI(
            centerX = centerX + dx,
            centerY = centerY + dy,
            roiWidth = roiWidth,
            roiHeight = roiHeight,
            rotation = rotation
        )
    }

    /**
     * Rotate ROI
     */
    fun rotate(degrees: Float): HandTrackingROI {
        return HandTrackingROI(
            centerX = centerX,
            centerY = centerY,
            roiWidth = roiWidth,
            roiHeight = roiHeight,
            rotation = rotation + degrees
        )
    }

    /**
     * Calculate Intersection over Union (IoU) with another ROI
     */
    fun iou(other: HandTrackingROI): Float {
        val box1 = getBoundingBox()
        val box2 = other.getBoundingBox()

        val x1 = maxOf(box1[0], box2[0])
        val y1 = maxOf(box1[1], box2[1])
        val x2 = minOf(box1[2], box2[2])
        val y2 = minOf(box1[3], box2[3])

        if (x2 < x1 || y2 < y1) return 0f

        val intersection = (x2 - x1) * (y2 - y1)
        val area1 = roiWidth * roiHeight
        val area2 = other.roiWidth * other.roiHeight
        val union = area1 + area2 - intersection

        return intersection / union
    }

    /**
     * Get area of ROI
     */
    fun getArea(): Float {
        return roiWidth * roiHeight
    }

    /**
     * Get aspect ratio (width / height)
     */
    fun getAspectRatio(): Float {
        return roiWidth / roiHeight
    }

    companion object {
        /**
         * Create ROI from bounding box [left, top, right, bottom]
         */
        fun fromBoundingBox(
            left: Float,
            top: Float,
            right: Float,
            bottom: Float,
            rotation: Float = 0f
        ): HandTrackingROI {
            val centerX = (left + right) / 2
            val centerY = (top + bottom) / 2
            val width = right - left
            val height = bottom - top

            return HandTrackingROI(centerX, centerY, width, height, rotation)
        }

        /**
         * Create ROI from detection result
         */
        fun fromDetection(
            box: FloatArray,
            expandFactor: Float = 1.5f
        ): HandTrackingROI {
            val roi = fromBoundingBox(box[0], box[1], box[2], box[3])
            return roi.expand(expandFactor)
        }

        /**
         * Create ROI from hand landmarks
         * @param landmarks Flat array of [x0, y0, z0, x1, y1, z1, ...]
         */
        fun fromLandmarks(
            landmarks: FloatArray,
            expandFactor: Float = 1.3f
        ): HandTrackingROI {
            var minX = Float.MAX_VALUE
            var minY = Float.MAX_VALUE
            var maxX = Float.MIN_VALUE
            var maxY = Float.MIN_VALUE

            // Find bounding box of landmarks
            for (i in 0 until landmarks.size / 3) {
                val x = landmarks[i * 3]
                val y = landmarks[i * 3 + 1]

                minX = minOf(minX, x)
                minY = minOf(minY, y)
                maxX = maxOf(maxX, x)
                maxY = maxOf(maxY, y)
            }

            val roi = fromBoundingBox(minX, minY, maxX, maxY)
            return roi.expand(expandFactor)
        }
    }

    override fun toString(): String {
        return "HandROI(center=[%.1f, %.1f], size=[%.1f x %.1f], rot=%.1f°)".format(
            centerX, centerY, roiWidth, roiHeight, rotation
        )
    }
}