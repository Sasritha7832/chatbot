import os
import json
import pickle
import random
import string
from flask import Flask, request, jsonify, render_template

import nltk
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Ensure NLTK resources are loaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

lemmatizer = WordNetLemmatizer()

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'chatbot_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'model', 'vectorizer.pkl')
INTENTS_PATH = os.path.join(BASE_DIR, 'data', 'intents.json')

# Global variables for models and data
model = None
vectorizer = None
intents_data = {}

def load_resources():
    """Loads the model, vectorizer, and intents JSON."""
    global model, vectorizer, intents_data
    
    print("Loading chatbot ML resources...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("WARNING: Model or Vectorizer pkl files not found! Please run train.py first.")
        return False
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)
        
    print("Resources loaded successfully.")
    return True

def preprocess_text(text):
    """
    Tokenizes, lowercases, removes punctuation, and lemmatizes the input text.
    Must match the preprocessing used in train.py.
    """
    tokens = nltk.word_tokenize(text.lower())
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in string.punctuation
    ]
    return " ".join(cleaned_tokens)

@app.route('/')
def home():
    """Serves the single-page frontend chat UI."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Accepts user message, predicts the intent,
    and returns a random response.
    """
    if model is None or vectorizer is None:
        # Try loading dynamically if they weren't ready at startup
        success = load_resources()
        if not success:
            return jsonify({
                "response": "Chatbot model is not trained or loaded. Please check the backend.",
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

    # Preprocess and vectorize the message
    processed_message = preprocess_text(user_message)
    features = vectorizer.transform([processed_message])
    
    # Predict probability distribution
    probabilities = model.predict_proba(features)[0]
    max_idx = int(probabilities.argmax())
    max_prob = float(probabilities[max_idx])
    predicted_tag = model.classes_[max_idx]
    
    # Confidence threshold fallback
    if max_prob < 0.4:
        return jsonify({
            "response": "I don't understand. Could you please rephrase or try another query?",
            "tag": "fallback",
            "confidence": max_prob
        })
        
    # Get random response corresponding to the predicted intent tag
    responses = None
    for intent in intents_data.get('intents', []):
        if intent['tag'] == predicted_tag:
            responses = intent['responses']
            break
            
    if responses:
        response_text = random.choice(responses)
    else:
        response_text = "I don't understand. Could you please rephrase or try another query?"
        
    return jsonify({
        "response": response_text,
        "tag": predicted_tag,
        "confidence": max_prob
    })

if __name__ == '__main__':
    # Load resources before running the server
    load_resources()
    app.run(host='127.0.0.1', port=5000, debug=True)
