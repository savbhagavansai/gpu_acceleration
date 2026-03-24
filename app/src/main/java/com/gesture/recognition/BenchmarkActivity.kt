package com.gesture.recognition

import android.os.Bundle
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Activity to run comprehensive model benchmarks
 *
 * Add to AndroidManifest.xml:
 * <activity android:name=".BenchmarkActivity" />
 *
 * Launch from MainActivity with:
 * startActivity(Intent(this, BenchmarkActivity::class.java))
 */
class BenchmarkActivity : AppCompatActivity() {

    private lateinit var textView: TextView
    private lateinit var scrollView: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Create simple UI programmatically
        scrollView = ScrollView(this).apply {
            layoutParams = android.view.ViewGroup.LayoutParams(
                android.view.ViewGroup.LayoutParams.MATCH_PARENT,
                android.view.ViewGroup.LayoutParams.MATCH_PARENT
            )
            setPadding(32, 32, 32, 32)
            setBackgroundColor(android.graphics.Color.BLACK)
        }

        textView = TextView(this).apply {
            typeface = android.graphics.Typeface.MONOSPACE
            textSize = 10f
            setTextColor(android.graphics.Color.GREEN)
            text = "Running comprehensive benchmark...\n\nThis may take 1-2 minutes.\nPlease wait...\n\n"
        }

        scrollView.addView(textView)
        setContentView(scrollView)

        // Run benchmark in background
        runBenchmark()
    }

    private fun runBenchmark() {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val benchmark = ModelBenchmark(applicationContext)

                // Run on background thread
                val report = withContext(Dispatchers.Default) {
                    benchmark.runCompleteBenchmark()
                }

                // Display results
                textView.text = report

                // Save to file
                FileLogger.i("BENCHMARK", report)

            } catch (e: Exception) {
                textView.text = "Benchmark failed:\n\n${e.message}\n\n${e.stackTraceToString()}"
            }
        }
    }
}