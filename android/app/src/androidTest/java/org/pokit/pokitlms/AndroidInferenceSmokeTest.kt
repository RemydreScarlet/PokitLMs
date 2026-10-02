package org.pokit.pokitlms

import android.os.ParcelFileDescriptor
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
        val modelPath = InstrumentationRegistry.getArguments().getString("pokitlms.modelPath")
        assumeTrue("set pokitlms.modelPath to run the full-model device check", !modelPath.isNullOrBlank())
        val model = File(modelPath!!)
        check(model.isFile) { "model file does not exist: $modelPath" }

        val bridge = NativeModelBridge()
        ParcelFileDescriptor.open(model, ParcelFileDescriptor.MODE_READ_ONLY).use { descriptor ->
            val handle = bridge.load(descriptor.fd)
            try {
                assertTrue(bridge.generate(handle, "hi", 1).isNotEmpty())
            } finally {
                bridge.close(handle)
            }
        }
    }

    @Test
    fun loadsBothGgufRunnersThroughJniAndGeneratesOnDevice() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val bridge = NativeModelBridge()
        for (filename in listOf("qwen35-smoke.gguf", "qwen3moe-smoke.gguf")) {
            val model = File(instrumentation.targetContext.cacheDir, filename)
            instrumentation.context.assets.open(filename).use { input ->
                model.outputStream().use { output -> input.copyTo(output) }
            }
            ParcelFileDescriptor.open(model, ParcelFileDescriptor.MODE_READ_ONLY).use { descriptor ->
                val handle = bridge.load(descriptor.fd)
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
