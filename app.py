import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import google.generativeai as genai

from rag import retrieve_context, ingest_documents

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure API
API_KEY = os.environ.get("GEMINI_API_KEY")
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

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles file uploads for the RAG knowledge base."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(DOCS_DIR, filename)
        file.save(file_path)
        
        # Trigger ingestion to update the vector store
        success = ingest_documents()
        
        if success:
            return jsonify({"success": True, "message": f"File '{filename}' successfully added to knowledge base!"})
        else:
            return jsonify({"error": "File saved, but embedding failed. Check API key."}), 500
            
    return jsonify({"error": "Invalid file type. Only TXT and PDF are allowed."}), 400

@app.route('/chat', methods=['POST'])
def chat():
    """
    Accepts user message, retrieves relevant context from RAG,
    and queries the Gemini LLM for a response.
    """
    if not API_KEY:
        return jsonify({
            "response": "ERROR: GEMINI_API_KEY is not set in the environment variables or .env file. Please add it to use the chatbot.",
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
        # Retrieve context from RAG engine
        context = retrieve_context(user_message)
        
        # Build prompt
        system_prompt = (
            "You are NexaBot, an intelligent, helpful, and friendly AI assistant. "
            "You format your responses beautifully using Markdown. "
        )
        
        if context:
            system_prompt += (
                "You have been provided with the following context documents to help answer the user's question. "
                "Use this context if it's relevant, but you can also answer general questions.\n\n"
                "=== CONTEXT START ===\n"
                f"{context}\n"
                "=== CONTEXT END ===\n"
            )

        # Initialize the model (using flash for speed and cost-effectiveness)
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=system_prompt
        )
        
        # Note: In a real app we'd pass the chat history to the API.
        # For simplicity and to match the old stateless backend (which relied on frontend history),
        # we'll just pass the current message.
        response = model.generate_content(user_message)
        
        return jsonify({
            "response": response.text,
            "tag": "gemini",
            "confidence": 1.0
        })
        
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            "response": f"Sorry, I encountered an error communicating with the AI: {str(e)}",
            "tag": "error",
            "confidence": 0.0
        }), 500

if __name__ == '__main__':
    # Initialize empty vector store if it doesn't exist
    if not os.path.exists(os.path.join(BASE_DIR, 'data', 'vector_store.pkl')):
        ingest_documents()
        
    app.run(host='127.0.0.1', port=5000, debug=True)
