package org.pokit.pokitlms

/** Small JNI boundary shared by the app and Android runtime smoke tests. */
internal class NativeModelBridge {
    external fun load(
        fd: Int, ioThreads: Int, useVulkan: Boolean = false,
        gpuTileMiB: Int = 8, gpuSubgroups: Boolean = true,
        gpuVectorizedQ4: Boolean = true
    ): Long
    external fun generate(handle: Long, prompt: String, maxTokens: Int): String
    external fun close(handle: Long)
    external fun verifyVulkan(): String

    companion object {
        init { System.loadLibrary("pokitlms_android") }
    }
}
