package org.pokit.pokitlms

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

class MainActivity : Activity() {
    private val native = NativeModelBridge()
    private val session = ModelSession(object : ModelSession.Backend {
        override fun load(fd: Int) = native.load(fd, 3)
        override fun generate(handle: Long, prompt: String, maxTokens: Int) = native.generate(handle, prompt, maxTokens)
        override fun close(handle: Long) = native.close(handle)
    })
    private var modelLoaded = false
    private lateinit var status: TextView
    private lateinit var prompt: EditText
    private lateinit var output: TextView
    private lateinit var loadButton: Button
    private lateinit var sendButton: Button

    companion object {
        private const val PICK_MODEL = 41
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 24, 20, 16)
        }
        status = TextView(this).apply { text = "GGUFモデルを選択してください"; textSize = 16f }
        loadButton = Button(this).apply {
            text = "GGUFを読み込む"
            setOnClickListener {
                val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE); type = "application/octet-stream"
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
                }
                startActivityForResult(intent, PICK_MODEL)
            }
        }
        prompt = EditText(this).apply { hint = "メッセージ"; minLines = 2; gravity = Gravity.TOP }
        sendButton = Button(this).apply {
            text = "送信"
            isEnabled = false
            setOnClickListener { generate() }
        }
        output = TextView(this).apply { textSize = 16f; setTextIsSelectable(true) }
        val scroll = ScrollView(this).apply { addView(output) }
        root.addView(status)
        root.addView(loadButton)
        root.addView(prompt, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(sendButton)
        root.addView(scroll, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)
    }

    @Deprecated("Activity result API retained for a dependency-free minimal app")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PICK_MODEL || resultCode != RESULT_OK) return
        val uri: Uri = data?.data ?: return
        try { contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) } catch (_: Exception) { }
        loadButton.isEnabled = false
        sendButton.isEnabled = false
        status.text = "モデルを確認中…"
        session.load(
            openSource = {
                val descriptor = contentResolver.openFileDescriptor(uri, "r") ?: error("ファイルを開けません")
                ModelSession.Source(descriptor.fd, descriptor)
            },
            onLoaded = { updateUi {
                modelLoaded = true
                status.text = "モデルを読み込みました"
                sendButton.isEnabled = true
                loadButton.isEnabled = true
            } },
            onError = { e -> updateUi {
                status.text = "読み込み失敗: ${e.message}"
                loadButton.isEnabled = true
                sendButton.isEnabled = modelLoaded
            } }
        )
    }

    private fun generate() {
        val text = prompt.text.toString().trim()
        if (!modelLoaded || text.isEmpty()) return
        sendButton.isEnabled = false
        loadButton.isEnabled = false
        output.append("\n\n> $text\n\n")
        session.generate(text, 256,
            onReply = { answer -> updateUi {
                output.append(answer)
                sendButton.isEnabled = true
                loadButton.isEnabled = true
            } },
            onError = { e -> updateUi {
                output.append("\n[エラー] ${e.message}")
                sendButton.isEnabled = true
                loadButton.isEnabled = true
            } }
        )
    }

    private fun updateUi(action: () -> Unit) {
        runOnUiThread { if (!isDestroyed) action() }
    }

    override fun onDestroy() {
        session.close()
        super.onDestroy()
    }
}
