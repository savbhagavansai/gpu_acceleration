package com.gesture.recognition

import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

/**
 * Smooths predictions by maintaining a history of recent predictions
 * Uses majority voting to reduce jitter
 * Matches Python's prediction_buffer behavior
 */
class PredictionSmoother(
    private val windowSize: Int = Config.PREDICTION_SMOOTHING_WINDOW
) {
    
    private val predictionHistory = mutableListOf<Int>()
    private val lock = ReentrantLock()
    
    /**
     * Add a new prediction to the history
     * 
     * @param classIdx Predicted class index
     */
    fun addPrediction(classIdx: Int) = lock.withLock {
        if (predictionHistory.size >= windowSize) {
            predictionHistory.removeAt(0)
        }
        predictionHistory.add(classIdx)
    }
    
    /**
     * Get smoothed prediction using majority voting
     * 
     * @return Most common class index in recent history, or null if history empty
     */
    fun getSmoothedPrediction(): Int? = lock.withLock {
        if (predictionHistory.isEmpty()) {
            return null
        }
        
        // Count occurrences of each class
        val counts = mutableMapOf<Int, Int>()
        for (classIdx in predictionHistory) {
            counts[classIdx] = counts.getOrDefault(classIdx, 0) + 1
        }
        
        // Return most frequent class
        counts.maxByOrNull { it.value }?.key
    }
    
    /**
     * Get smoothed prediction with confidence
     * Confidence = (count of most common class) / (total predictions)
     * 
     * @return Pair of (class_index, confidence) or null if history empty
     */
    fun getSmoothedPredictionWithConfidence(): Pair<Int, Float>? = lock.withLock {
        if (predictionHistory.isEmpty()) {
            return null
        }
        
        // Count occurrences
        val counts = mutableMapOf<Int, Int>()
        for (classIdx in predictionHistory) {
            counts[classIdx] = counts.getOrDefault(classIdx, 0) + 1
        }
        
        // Find most frequent
        val (mostCommonClass, count) = counts.maxByOrNull { it.value } ?: return null
        
        // Calculate confidence as proportion
        val confidence = count.toFloat() / predictionHistory.size.toFloat()
        
        Pair(mostCommonClass, confidence)
    }
    
    /**
     * Check if prediction is stable
     * Stable = all recent predictions are the same
     * 
     * @return True if all predictions in history are identical
     */
    fun isStable(): Boolean = lock.withLock {
        if (predictionHistory.isEmpty()) return false
        if (predictionHistory.size < windowSize) return false
        
        val firstPrediction = predictionHistory[0]
        predictionHistory.all { it == firstPrediction }
    }
    
    /**
     * Get current history size
     */
    fun size(): Int = lock.withLock {
        predictionHistory.size
    }
    
    /**
     * Clear prediction history
     */
    fun clear() = lock.withLock {
        predictionHistory.clear()
    }
    
    /**
     * Get distribution of predictions in history
     * 
     * @return Map of class_index -> count
     */
    fun getDistribution(): Map<Int, Int> = lock.withLock {
        val counts = mutableMapOf<Int, Int>()
        for (classIdx in predictionHistory) {
            counts[classIdx] = counts.getOrDefault(classIdx, 0) + 1
        }
        counts.toMap()
    }
    
    override fun toString(): String = lock.withLock {
        "PredictionSmoother(size=${predictionHistory.size}/$windowSize, stable=${isStable()})"
    }
}
