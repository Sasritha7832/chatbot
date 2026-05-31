import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai

from rag import retrieve_context, ingest_documents

# Load environment variables from .env (only works locally; on Render use dashboard env vars)
load_dotenv()

app = Flask(__name__)

# Configure API at startup
API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if API_KEY:
    genai.configure(api_key=API_KEY)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, 'data', 'docs')

# Ensure docs directory exists
os.makedirs(DOCS_DIR, exist_ok=True)

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    """Serves the single-page frontend chat UI."""
    return render_template('index.html')


@app.route('/health')
def health():
    """
    Quick diagnostic endpoint.
    Visit /health on your deployed URL to check configuration.
    """
    key_set = bool(API_KEY)
    key_preview = (API_KEY[:8] + "...") if key_set else "NOT SET"
    return jsonify({
        "status": "ok",
        "api_key_configured": key_set,
        "api_key_preview": key_preview,
        "docs_dir_exists": os.path.isdir(DOCS_DIR),
    })


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads for the RAG knowledge base."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DOCS_DIR, filename)
        file.save(file_path)

        # Trigger ingestion to update the vector store
        success, error_msg = ingest_documents()

        if success:
            return jsonify({
                "success": True,
                "message": f"✅ '{filename}' added to the knowledge base!"
            })
        else:
            # Surface the real error to the frontend
            return jsonify({
                "error": f"File saved, but embedding failed: {error_msg}"
            }), 500

    return jsonify({"error": "Invalid file type. Only TXT and PDF are supported."}), 400


@app.route('/chat', methods=['POST'])
def chat():
    """
    Accepts a user message, retrieves relevant context from RAG,
    and queries the Gemini LLM for a response.
    """
    # Re-read the key dynamically so hot-set env vars work without restart
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({
            "response": (
                "⚠️ **Configuration Error**: `GEMINI_API_KEY` is not set on this server.\n\n"
                "If you're using Render, go to your Web Service → **Environment** tab and add "
                "`GEMINI_API_KEY` with your Google AI Studio key."
            ),
            "tag": "error",
            "confidence": 0.0
        }), 500

    genai.configure(api_key=api_key)

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
        # Retrieve context from RAG engine (best-effort; falls back to empty)
        context = retrieve_context(user_message)

        system_prompt = (
            "You are NexaBot, an intelligent, helpful, and friendly AI assistant. "
            "You format your responses beautifully using Markdown (headers, bullet points, code blocks, etc.)."
        )

        if context:
            system_prompt += (
                "\n\nYou have been given context documents to help answer the user's question. "
                "Use this context when relevant, and clearly mention the source.\n\n"
                "=== CONTEXT START ===\n"
                f"{context}\n"
                "=== CONTEXT END ==="
            )

        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=system_prompt
        )

        response = model.generate_content(user_message)

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
                "Please check that your Gemini API key is valid and has not expired."
            ),
            "tag": "error",
            "confidence": 0.0
        }), 500


if __name__ == '__main__':
    # Initialize empty vector store if it doesn't exist
    if not os.path.exists(os.path.join(BASE_DIR, 'data', 'vector_store.pkl')):
        ingest_documents()

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
