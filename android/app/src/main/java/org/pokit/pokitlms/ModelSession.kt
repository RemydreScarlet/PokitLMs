package org.pokit.pokitlms

import java.io.Closeable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** Owns native handles and descriptors exclusively on one background worker. */
internal class ModelSession(
    private val backend: Backend,
    private val worker: ExecutorService = Executors.newSingleThreadExecutor()
) : Closeable {
    interface Backend {
        fun load(fd: Int): Long
        fun generate(handle: Long, prompt: String, maxTokens: Int): String
        fun close(handle: Long)
    }

    data class Source(val fd: Int, val resource: Closeable)

    private val closed = AtomicBoolean(false)
    private val queueLock = Any()
    // These fields are read and written only on the worker.
    private var handle = 0L
    private var source: Source? = null

    private fun submit(operation: () -> Unit) {
        synchronized(queueLock) {
            if (!closed.get()) worker.execute(operation)
        }
    }

    fun load(openSource: () -> Source, onLoaded: () -> Unit, onError: (Exception) -> Unit) {
        submit {
            if (closed.get()) return@submit
            var candidate: Source? = null
            var candidateHandle = 0L
            try {
                candidate = openSource()
                candidateHandle = backend.load(candidate.fd)
                check(candidateHandle != 0L) { "モデルを読み込めませんでした" }
                if (closed.get()) return@submit
                releaseCurrent()
                source = candidate
                handle = candidateHandle
                candidate = null
                candidateHandle = 0L
                if (!closed.get()) onLoaded()
            } catch (e: Exception) {
                if (!closed.get()) onError(e)
            } finally {
                try {
                    if (candidateHandle != 0L) backend.close(candidateHandle)
                } finally {
                    candidate?.resource?.close()
                }
            }
        }
    }

    fun generate(prompt: String, maxTokens: Int, onReply: (String) -> Unit, onError: (Exception) -> Unit) {
        submit {
            // Check the current owner and handle here, after all prior queued operations.
            if (closed.get()) return@submit
            try {
                check(handle != 0L) { "モデルが読み込まれていません" }
                val answer = backend.generate(handle, prompt, maxTokens)
                if (!closed.get()) onReply(answer)
            } catch (e: Exception) {
                if (!closed.get()) onError(e)
            }
        }
    }

    private fun releaseCurrent() {
        val previousHandle = handle
        val previousSource = source
        handle = 0L
        source = null
        try {
            if (previousHandle != 0L) backend.close(previousHandle)
        } finally {
            previousSource?.resource?.close()
        }
    }

    override fun close() {
        synchronized(queueLock) {
            if (!closed.compareAndSet(false, true)) return
            // Cleanup runs after any native call already in progress; close never waits for it.
            worker.execute { releaseCurrent() }
            worker.shutdown()
        }
    }
}
