import os
import json
import pickle
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Download necessary NLTK datasets
import nltk
print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """
    Tokenizes, lowercases, removes punctuation, and lemmatizes the input text.
    """
    tokens = nltk.word_tokenize(text.lower())
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in string.punctuation
    ]
    return " ".join(cleaned_tokens)

def main():
    # Paths setup
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intents_path = os.path.join(base_dir, 'data', 'intents.json')
    model_dir = os.path.join(base_dir, 'model')
    
    os.makedirs(model_dir, exist_ok=True)
    
    # 2. Load intents.json
    print(f"Loading training data from {intents_path}...")
    with open(intents_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract patterns and tags
    patterns = []
    tags = []
    
    for intent in data['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(tag)
            
    print(f"Loaded {len(patterns)} patterns across {len(set(tags))} intents.")
    
    # Preprocess all patterns
    preprocessed_patterns = [preprocess_text(p) for p in patterns]
    
    # 3. Train/Test Split (80/20) - Stratified to ensure all classes are represented in train/test
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        preprocessed_patterns, 
        tags, 
        test_size=0.2, 
        random_state=42, 
        stratify=tags
    )
    
    # 4. TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    # 5. Train Logistic Regression model
    model = LogisticRegression(C=10.0, max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluate accuracy on test split
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 7. Retrain on the entire dataset for deployment robustness
    print("Training final model on full dataset...")
    final_vectorizer = TfidfVectorizer()
    X_full = final_vectorizer.fit_transform(preprocessed_patterns)
    
    final_model = LogisticRegression(C=10.0, max_iter=200, random_state=42)
    final_model.fit(X_full, tags)
    
    # 8. Save model and vectorizer
    model_path = os.path.join(model_dir, 'chatbot_model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    print(f"Saving final model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)
        
    print(f"Saving final vectorizer to {vectorizer_path}...")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(final_vectorizer, f)
        
    print("Model training pipeline completed successfully!")

if __name__ == '__main__':
    main()
