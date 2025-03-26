import numpy as np
import re
import torch
import joblib
import torch.nn as nn
from difflib import SequenceMatcher
from urllib.parse import urlparse

# Load trained models and vectorizer
rf_model = joblib.load("random_forest_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=8, hidden_dim=16):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Use last LSTM output
        return x

# Load the LSTM model
vocab_size = len(vectorizer.get_feature_names_out()) + 1
lstm_model = LSTMModel(vocab_size)
lstm_model.load_state_dict(torch.load("lstm_model.pth"))
lstm_model.eval()

# ✅ Feature Extraction (Using Trigrams)
def extract_ngrams(url, n=3):
    url = re.sub(r'[^a-zA-Z0-9]', '', url.lower())  # Remove special characters
    return ["".join(url[i:i+n]) for i in range(len(url)-n+1)]

word_to_index = {word: idx + 1 for idx, word in enumerate(vectorizer.get_feature_names_out())}

# Convert URL text to numerical sequence
def text_to_sequence(text, max_len=20):
    ngrams = extract_ngrams(text, 3)  # Extract trigrams
    seq = [word_to_index.get(ngram, 0) for ngram in ngrams]
    return seq[:max_len] + [0] * (max_len - len(seq))  # Pad sequence

# ✅ Improved Typo-Squatting Detection
def is_typo_squatting(domain, safe_domains, threshold=0.75):
    if domain in safe_domains:
        return False  # If it's a known safe domain, it's not phishing
    for safe_domain in safe_domains:
        similarity = SequenceMatcher(None, domain, safe_domain).ratio()
        if similarity > threshold:
            return True  # Looks like a phishing attempt
    return False

# ✅ Improved Domain Extraction
def get_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    domain = domain.replace("www.", "")  # Remove 'www.'
    return domain

# ✅ Improved Prediction Function
def predict_url(url):
    domain = get_domain(url)
    bigram_features = " ".join(extract_ngrams(url, 3))  # Using trigrams

    # Random Forest Prediction
    rf_features = vectorizer.transform([bigram_features])
    rf_prob = rf_model.predict_proba(rf_features)[:, 1][0]

    # LSTM Prediction
    lstm_input = np.array([text_to_sequence(bigram_features)])
    lstm_tensor = torch.tensor(lstm_input.tolist(), dtype=torch.long)
    with torch.no_grad():
        lstm_output = lstm_model(lstm_tensor)
        lstm_prob = torch.softmax(lstm_output.to(torch.float32), dim=1)[:, 1].item()

    # Safe domains
    safe_domains = ["google.com", "paypal.com", "facebook.com"]
    is_phishing_typo = is_typo_squatting(domain, safe_domains)

    # ✅ Fix: If domain is safe, force prediction to benign
    if domain in safe_domains:
        return "Benign", 0.01  # Set low confidence to avoid false positives

    # ✅ Ensemble Prediction
    final_prob = (rf_prob + lstm_prob) / 2
    if final_prob > 0.5 or is_phishing_typo:
        return "Phishing", final_prob
    else:
        return "Benign", final_prob

# ✅ Test Cases
urls_to_test = [
    "https://www.google.com/",   # Safe
    "http://g00gle.com",         # Phishing
    "http://paypal.com",
    "http://facebo0k.com",       # Phishing
    "https://secure-paypal.com",
     "http://p@ypal.com" # Phishing
]

for url in urls_to_test:
    prediction, probability = predict_url(url)
    print(f"URL: {url} → Prediction: {prediction} (Confidence: {probability:.2f})")