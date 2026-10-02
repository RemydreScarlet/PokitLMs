package org.pokit.pokitlms

import android.os.ParcelFileDescriptor
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class AndroidInferenceSmokeTest {
    @Test
    fun loadsCallerProvidedFullModelWhenConfigured() {
        val arguments = InstrumentationRegistry.getArguments()
        val modelPath = arguments.getString("pokitlms.modelPath")
        assumeTrue("set pokitlms.modelPath to run the full-model device check", !modelPath.isNullOrBlank())
        val ioThreads = arguments.getString("pokitlms.ioThreads")?.toIntOrNull() ?: 3
        val maxTokens = arguments.getString("pokitlms.maxTokens")?.toIntOrNull() ?: 1
        val model = File(modelPath!!)
        check(model.isFile) { "model file does not exist: $modelPath" }

        val bridge = NativeModelBridge()
        ParcelFileDescriptor.open(model, ParcelFileDescriptor.MODE_READ_ONLY).use { descriptor ->
            val handle = bridge.load(descriptor.fd, ioThreads)
            try {
                val start = SystemClock.elapsedRealtimeNanos()
                val reply = bridge.generate(handle, "hi", maxTokens)
                val elapsedMs = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000
                Log.i("PokitLMsAB", "io_threads=$ioThreads max_tokens=$maxTokens elapsed_ms=$elapsedMs reply=$reply")
                assertTrue(reply.isNotEmpty())
            } finally {
                bridge.close(handle)
            }
        }
    }

    @Test
    fun loadsBothGgufRunnersThroughJniAndGeneratesOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val bridge = NativeModelBridge()
        for (filename in listOf("qwen35-smoke.gguf", "qwen35moe-smoke.gguf", "qwen3moe-smoke.gguf")) {
            val model = File(instrumentation.targetContext.cacheDir, filename)
            instrumentation.context.assets.open(filename).use { input ->
                model.outputStream().use { output -> input.copyTo(output) }
            }
            ParcelFileDescriptor.open(model, ParcelFileDescriptor.MODE_READ_ONLY).use { descriptor ->
            val handle = bridge.load(descriptor.fd, 3)
                try {
                    assertEquals(filename, "aaa", bridge.generate(handle, "hi", 3))
                } finally {
                    bridge.close(handle)
                }
            }
            check(model.delete() || !model.exists())
        }
    }
}
