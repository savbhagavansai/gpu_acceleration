package com.gesture.recognition

/**
 * Configuration object matching Python config.py
 * Contains all constants for gesture recognition pipeline
 */
object Config {
    
    // ==================== SEQUENCE PARAMETERS ====================
    const val SEQUENCE_LENGTH = 15  // Number of frames per sequence
    const val TARGET_FPS = 12       // Target frame rate for normalization
    
    // ==================== MEDIAPIPE PARAMETERS ====================
    const val MP_HANDS_CONFIDENCE = 0.5f
    const val MP_HANDS_TRACKING_CONFIDENCE = 0.5f
    
    // ==================== NORMALIZATION PARAMETERS ====================
    const val NORMALIZATION_CLIP_RANGE = 2.0f  // Clip outliers to [-2.0, 2.0]
    const val MIN_HAND_SCALE = 0.01f           // Minimum hand scale to avoid division by zero
    
    // ==================== MODEL ARCHITECTURE ====================
    const val NUM_FEATURES = 63     // 21 hand landmarks × 3 coordinates
    const val NUM_CLASSES = 11      // Number of gesture classes
    
    // ==================== MODEL FILE ====================
    const val ONNX_MODEL_FILENAME = "gesture_model.onnx"

    // ==================== INFERENCE PARAMETERS ====================
    const val CONFIDENCE_THRESHOLD = 0.6f  // Minimum confidence for prediction
    const val PREDICTION_SMOOTHING_WINDOW = 5  // Last N predictions for smoothing

    // ==================== LABEL MAPPING ====================
    val LABEL_TO_IDX = mapOf(
        "doing_other_things" to 0,
        "swipe_left" to 1,
        "swipe_right" to 2,
        "thumb_down" to 3,
        "thumb_up" to 4,
        "v_gesture" to 5,
        "top" to 6,
        "left_gesture" to 7,
        "right_gesture" to 8,
        "stop_sign" to 9,
        "heart" to 10
    )

    val IDX_TO_LABEL = LABEL_TO_IDX.entries.associate { (k, v) -> v to k }

    // ==================== UI COLORS ====================
    object Colors {
        const val SUCCESS = 0xFF00FF00.toInt()      // Green
        const val WARNING = 0xFF00FFFF.toInt()      // Yellow
        const val ERROR = 0xFF0000FF.toInt()        // Red
        const val INFO = 0xFFFFFFFF.toInt()         // White
        const val BACKGROUND = 0xFF000000.toInt()   // Black
        const val ORANGE = 0xFFFFA500.toInt()       // Orange
    }

    // ==================== CAMERA PARAMETERS ====================
    const val CAMERA_WIDTH = 1280
    const val CAMERA_HEIGHT = 720
    const val CAMERA_FPS = 30

    // ==================== FPS TRACKING ====================
    const val FPS_BUFFER_SIZE = 30  // Average FPS over last 30 frames
}