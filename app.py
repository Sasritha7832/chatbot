import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag import retrieve_context, ingest_documents

# Load .env locally (on Render, env vars are set via the dashboard)
load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'data', 'docs')
os.makedirs(DOCS_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_client():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health')
def health():
    """Diagnostic endpoint — visit /health to confirm the key is set."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    key_set = bool(api_key)
    return jsonify({
        "status": "ok",
        "api_key_configured": key_set,
        "api_key_preview": (api_key[:10] + "...") if key_set else "NOT SET",
        "docs_dir_exists": os.path.isdir(DOCS_DIR),
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DOCS_DIR, filename)
        file.save(file_path)

        success, error_msg = ingest_documents()

        if success:
            return jsonify({
                "success": True,
                "message": f"✅ '{filename}' added to the knowledge base! You can now ask questions about it."
            })
        else:
            return jsonify({
                "error": f"File saved, but embedding failed: {error_msg}"
            }), 500

    return jsonify({"error": "Invalid file type. Only TXT and PDF are supported."}), 400


@app.route('/chat', methods=['POST'])
def chat():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({
            "response": (
                "⚠️ **Configuration Error**: `GEMINI_API_KEY` is not set on this server.\n\n"
                "Go to your Render Web Service → **Environment** tab and add your key."
            ),
            "tag": "error",
            "confidence": 0.0
        }), 500

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    user_message = data['message'].strip()
    if not user_message:
        return jsonify({
            "response": "I see you didn't type anything. What's on your mind?",
            "tag": "empty",
            "confidence": 1.0
        })

    try:
        client = genai.Client(api_key=api_key)

        # Retrieve RAG context (best-effort)
        context = retrieve_context(user_message)

        system_prompt = (
            "You are NexaBot, an intelligent, helpful, and friendly AI assistant. "
            "Format your responses beautifully using Markdown (headers, bullet points, code blocks, bold text, etc.)."
        )

        if context:
            system_prompt += (
                "\n\nYou have been given context from uploaded documents. "
                "Use this context when relevant and mention the source.\n\n"
                "=== CONTEXT START ===\n"
                f"{context}\n"
                "=== CONTEXT END ==="
            )

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
        )

        return jsonify({
            "response": response.text,
            "tag": "gemini",
            "confidence": 1.0
        })

    except Exception as e:
        error_str = str(e)
        print(f"Chat error: {error_str}")
        return jsonify({
            "response": (
                f"⚠️ **AI Error**: {error_str}\n\n"
                "Please check that your Gemini API key is valid."
            ),
            "tag": "error",
            "confidence": 0.0
        }), 500


if __name__ == '__main__':
    if not os.path.exists(os.path.join(BASE_DIR, 'data', 'vector_store.pkl')):
        ingest_documents()

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
