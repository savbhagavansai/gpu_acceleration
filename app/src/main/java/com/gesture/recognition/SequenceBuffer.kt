package com.gesture.recognition

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Thread-safe circular buffer for storing landmark sequences
 * Matches Python's deque(maxlen=SEQUENCE_LENGTH) behavior
 */
class SequenceBuffer(private val maxLength: Int = Config.SEQUENCE_LENGTH) {
    
    private val buffer = mutableListOf<FloatArray>()
    private val lock = ReentrantLock()
    
    /**
     * Add a frame to the buffer
     * If buffer is full, oldest frame is removed
     * 
     * @param landmarks Normalized landmarks (63 features)
     */
    fun add(landmarks: FloatArray) = lock.withLock {
        require(landmarks.size == Config.NUM_FEATURES) {
            "Expected ${Config.NUM_FEATURES} features, got ${landmarks.size}"
        }
        
        if (buffer.size >= maxLength) {
            buffer.removeAt(0)  // Remove oldest
        }
        buffer.add(landmarks)
    }
    
    /**
     * Get current buffer size
     * 
     * @return Number of frames in buffer
     */
    fun size(): Int = lock.withLock {
        buffer.size
    }
    
    /**
     * Check if buffer is full
     * 
     * @return True if buffer has maxLength frames
     */
    fun isFull(): Boolean = lock.withLock {
        buffer.size == maxLength
    }
    
    /**
     * Get buffer as 2D array for model input
     * Returns shape: [sequence_length, num_features]
     * 
     * @return 2D FloatArray or null if buffer not full
     */
    fun getSequence(): Array<FloatArray>? = lock.withLock {
        if (buffer.size != maxLength) {
            return null
        }
        
        // Deep copy to avoid concurrent modification
        Array(maxLength) { i ->
            buffer[i].copyOf()
        }
    }
    
    /**
     * Get buffer as flattened array for model input
     * Returns shape: [sequence_length * num_features]
     * 
     * @return Flattened FloatArray or null if buffer not full
     */
    fun getSequenceFlattened(): FloatArray? = lock.withLock {
        if (buffer.size != maxLength) {
            return null
        }
        
        val flattened = FloatArray(maxLength * Config.NUM_FEATURES)
        var idx = 0
        for (frame in buffer) {
            for (value in frame) {
                flattened[idx++] = value
            }
        }
        flattened
    }
    
    /**
     * Clear the buffer
     */
    fun clear() = lock.withLock {
        buffer.clear()
    }
    
    /**
     * Get current buffer contents (for debugging)
     * 
     * @return Copy of current buffer
     */
    fun getBufferCopy(): List<FloatArray> = lock.withLock {
        buffer.map { it.copyOf() }
    }
    
    override fun toString(): String = lock.withLock {
        "SequenceBuffer(size=${buffer.size}/$maxLength, full=${isFull()})"
    }
}
