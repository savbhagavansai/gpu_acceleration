package com.gesture.recognition

import android.content.Context
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*

/**
 * File-based logger for debugging without USB/logcat access
 *
 * Logs are saved to: /storage/emulated/0/Android/data/com.gesture.recognition/files/debug_log.txt
 *
 * Usage:
 *   FileLogger.init(context)
 *   FileLogger.d("TAG", "message")
 *   FileLogger.e("TAG", "error message", exception)
 */
object FileLogger {

    private var logFile: FileWriter? = null
    private var isInitialized = false
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    /**
     * Initialize the logger - call this in MainActivity.onCreate()
     */
    fun init(context: Context) {
        if (isInitialized) return

        try {
            // Save to Downloads folder - MUCH easier to access!
            val downloadsDir = android.os.Environment.getExternalStoragePublicDirectory(
                android.os.Environment.DIRECTORY_DOWNLOADS
            )

            // Create subdirectory for organization
            val logDir = File(downloadsDir, "GestureRecognition")
            if (!logDir.exists()) {
                logDir.mkdirs()
                android.util.Log.d("FileLogger", "Created log directory: ${logDir.absolutePath}")
            }

            val file = File(logDir, "debug_log.txt")

            // Open in append mode
            logFile = FileWriter(file, true)

            isInitialized = true

            // Write header
            val separator = "═".repeat(60)
            writeRaw("\n$separator")
            writeRaw("APP STARTED: ${dateFormat.format(Date())}")
            writeRaw("Log file: ${file.absolutePath}")
            writeRaw(separator)

        } catch (e: Exception) {
            e.printStackTrace()
            android.util.Log.e("FileLogger", "Failed to initialize FileLogger", e)
        }
    }

    /**
     * Debug log
     */
    fun d(tag: String, message: String) {
        write("D", tag, message)
    }

    /**
     * Info log
     */
    fun i(tag: String, message: String) {
        write("I", tag, message)
    }

    /**
     * Warning log
     */
    fun w(tag: String, message: String) {
        write("W", tag, message)
    }

    /**
     * Error log
     */
    fun e(tag: String, message: String, throwable: Throwable? = null) {
        write("E", tag, message)
        throwable?.let {
            writeRaw("  Exception: ${it.javaClass.simpleName}: ${it.message}")
            it.stackTrace.take(5).forEach { element ->
                writeRaw("    at $element")
            }
        }
    }

    /**
     * Write formatted log entry
     */
    private fun write(level: String, tag: String, message: String) {
        if (!isInitialized) return

        try {
            val timestamp = dateFormat.format(Date())
            val logLine = "$timestamp [$level] $tag: $message"
            logFile?.write("$logLine\n")
            logFile?.flush()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Write raw line (no formatting)
     */
    private fun writeRaw(message: String) {
        if (!isInitialized) return

        try {
            logFile?.write("$message\n")
            logFile?.flush()
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    /**
     * Add separator line
     */
    fun separator() {
        writeRaw("─".repeat(60))
    }

    /**
     * Add section header
     */
    fun section(title: String) {
        val separator = "═".repeat(60)
        writeRaw("\n$separator")
        writeRaw(" $title")
        writeRaw(separator)
    }

    /**
     * Close the logger - call in onDestroy()
     */
    fun close() {
        try {
            writeRaw("\n${"═".repeat(60)}")
            writeRaw("APP STOPPED: ${dateFormat.format(Date())}")
            writeRaw("${"═".repeat(60)}\n")

            logFile?.close()
            logFile = null
            isInitialized = false
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }
}