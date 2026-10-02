package org.pokit.pokitlms

/** Small JNI boundary shared by the app and Android runtime smoke tests. */
internal class NativeModelBridge {
    external fun load(fd: Int): Long
    external fun generate(handle: Long, prompt: String, maxTokens: Int): String
    external fun close(handle: Long)

    companion object {
        init { System.loadLibrary("pokitlms_android") }
    }
}
