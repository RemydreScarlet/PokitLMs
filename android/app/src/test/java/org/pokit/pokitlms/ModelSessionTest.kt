package org.pokit.pokitlms

import org.junit.Assert.*
import org.junit.Test
import java.io.Closeable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class ModelSessionTest {
    private class Resource : Closeable {
        val closes = AtomicInteger()
        override fun close() { closes.incrementAndGet() }
    }

    private open class Backend : ModelSession.Backend {
        val closes = AtomicInteger()
        val generations = AtomicInteger()
        override fun load(fd: Int) = 123L
        override fun generate(handle: Long, prompt: String, maxTokens: Int): String {
            generations.incrementAndGet()
            check(handle == 123L && closes.get() == 0) { "use after native close" }
            return "reply"
        }
        override fun close(handle: Long) {
            check(handle == 123L)
            closes.incrementAndGet()
        }
    }

    private fun CountDownLatch.awaitReady() = assertTrue("worker timed out", await(5, TimeUnit.SECONDS))

    @Test(timeout = 10000)
    fun destructionSkipsGenerationQueuedBeforeCleanup() {
        val worker = Executors.newSingleThreadExecutor()
        val backend = Backend()
        val resource = Resource()
        val session = ModelSession(backend, worker)
        val gate = CountDownLatch(1)
        try {
            val loaded = CountDownLatch(1)
            session.load({ ModelSession.Source(9, resource) }, { loaded.countDown() }, { throw it })
            loaded.awaitReady()
            val waiting = CountDownLatch(1)
            worker.execute { waiting.countDown(); gate.awaitReady() }
            waiting.awaitReady()
            session.generate("hello", 256, { fail("reply after close") }, { throw it })
            session.close()
            session.close()
            gate.countDown()
            assertTrue(worker.awaitTermination(5, TimeUnit.SECONDS))
            assertEquals(0, backend.generations.get())
            assertEquals(1, backend.closes.get())
            assertEquals(1, resource.closes.get())
        } finally {
            gate.countDown()
            session.close()
            worker.shutdownNow()
        }
    }

    @Test(timeout = 10000)
    fun closeReturnsWhileInferenceIsStillRunning() {
        val entered = CountDownLatch(1)
        val gate = CountDownLatch(1)
        val replies = AtomicInteger()
        val backend = object : Backend() {
            override fun generate(handle: Long, prompt: String, maxTokens: Int): String {
                entered.countDown()
                gate.awaitReady()
                return super.generate(handle, prompt, maxTokens)
            }
        }
        val resource = Resource()
        val worker = Executors.newSingleThreadExecutor()
        val session = ModelSession(backend, worker)
        try {
            session.load({ ModelSession.Source(9, resource) }, {}, { throw it })
            session.generate("hello", 256, { replies.incrementAndGet() }, { throw it })
            entered.awaitReady()
            session.close()
            // Cleanup must wait on the worker, while close itself has already returned.
            assertEquals(0, backend.closes.get())
            assertEquals(0, resource.closes.get())
            gate.countDown()
            assertTrue(worker.awaitTermination(5, TimeUnit.SECONDS))
            assertEquals(1, backend.generations.get())
            assertEquals(1, backend.closes.get())
            assertEquals(1, resource.closes.get())
            assertEquals(0, replies.get())
        } finally {
            gate.countDown()
            session.close()
            worker.shutdownNow()
        }
    }

    @Test(timeout = 10000)
    fun modelLoadedAfterDestructionIsDisposed() {
        val entered = CountDownLatch(1)
        val gate = CountDownLatch(1)
        val callbacks = AtomicInteger()
        val backend = object : Backend() {
            override fun load(fd: Int): Long {
                entered.countDown()
                gate.awaitReady()
                return super.load(fd)
            }
        }
        val resource = Resource()
        val worker = Executors.newSingleThreadExecutor()
        val session = ModelSession(backend, worker)
        try {
            session.load({ ModelSession.Source(9, resource) }, { callbacks.incrementAndGet() }, { throw it })
            entered.awaitReady()
            session.close()
            gate.countDown()
            assertTrue(worker.awaitTermination(5, TimeUnit.SECONDS))
            assertEquals(1, backend.closes.get())
            assertEquals(1, resource.closes.get())
            assertEquals(0, callbacks.get())
        } finally {
            gate.countDown()
            session.close()
            worker.shutdownNow()
        }
    }

    @Test(timeout = 10000)
    fun failedLoadClosesItsDescriptor() {
        val worker = Executors.newSingleThreadExecutor()
        val backend = object : Backend() {
            override fun load(fd: Int): Long = throw IllegalArgumentException("bad GGUF")
        }
        val resource = Resource()
        val session = ModelSession(backend, worker)
        try {
            val failed = CountDownLatch(1)
            session.load({ ModelSession.Source(9, resource) }, { fail("load must fail") }, { failed.countDown() })
            failed.awaitReady()
            session.close()
            assertTrue(worker.awaitTermination(5, TimeUnit.SECONDS))
            assertEquals(0, backend.closes.get())
            assertEquals(1, resource.closes.get())
        } finally {
            session.close()
            worker.shutdownNow()
        }
    }
}
