# NexaBot — Smart AI Chat Assistant

A modern, online web-based chatbot application built with Python, Flask, `scikit-learn`, and `nltk`. It classifies user messages using a Machine Learning classifier (TF-IDF + Logistic Regression) and serves a premium glassmorphic chat interface.

## Stack Overview
- **Backend:** Python + Flask
- **NLP Preprocessing:** NLTK (Tokenization, Punctuation removal, WordNet Lemmatization)
- **Machine Learning:** scikit-learn (`TfidfVectorizer` + `LogisticRegression` with `max_iter=200`)
- **Frontend:** Vanilla HTML, CSS, and JS (incorporating responsive grid, local storage caching, typing animation, avatars, and timestamps)
- **Data storage:** `intents.json` dataset (15 intents, multiple patterns and responses)

---

## Project Structure
```
chatbotapp/
├── data/
│   └── intents.json          # Training dataset
├── model/
│   ├── train.py              # Model training pipeline
│   ├── chatbot_model.pkl     # Trained Logistic Regression classifier (Generated)
│   └── vectorizer.pkl        # Trained TF-IDF Vectorizer (Generated)
├── templates/
│   └── index.html            # Premium Glassmorphic Chat UI
├── app.py                    # Flask server
├── requirements.txt          # Python dependencies
└── README.md                 # Setup instructions (This file)
```

---

## Setup and Execution Guide

Follow these steps to set up and run NexaBot locally:

### 1. Create a Virtual Environment
Initialize a clean Python virtual environment to manage dependencies:
```bash
python -m venv venv
```

### 2. Activate the Virtual Environment
Activate the environment based on your operating system:
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning Model
Run the training script to update the model with the latest intents. This script handles NLTK resource downloads (`punkt`, `wordnet`, `omw-1.4`), cleans and lemmatizes the text data, performs an 80/20 train-test split for evaluation, and outputs the serialized model files (`.pkl`):
```bash
python model/train.py
```

### 5. Run the Flask Web Application
Start the Flask web server:
```bash
python app.py
```

### 6. Chat with NexaBot
Open your web browser and navigate to:
```
http://localhost:5000
```

---

## How It Works Under the Hood

1. **Text Preprocessing:** When you send a message, it is lowercased, tokenized via NLTK's `word_tokenize`, stripped of punctuation, and lemmatized via `WordNetLemmatizer`. This reduces variations of words (e.g. "running" and "runs" both become "run").
2. **Feature Extraction:** The text is transformed into numerical vectors using a saved `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) model.
3. **Intent Prediction:** The vectorized message is passed through a `LogisticRegression` model, which computes class probability distributions.
4. **Threshold Filtering:** 
   * If the highest class probability is **less than 40% (0.40)**, the bot responds with a fallback: *"I don't understand. Could you please rephrase or try another query?"*.
   * If the probability is **40% or higher**, the bot selects a random response from the matches corresponding to the predicted intent in `data/intents.json`.
5. **Frontend State Caching:** The frontend saves your chat history in `localStorage`, so your conversation remains intact when refreshing the browser page. Use the **Clear Chat** button on the sidebar to reset the session.
