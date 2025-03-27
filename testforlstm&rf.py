import numpy as np
import pandas as pd
import re
import torch
import joblib
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class PhishingDetector:
    def __init__(self, rf_model_path, lstm_model_path, vectorizer_path):
        self.rf_model = joblib.load(rf_model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.vocab_size = len(self.vectorizer.get_feature_names_out()) + 1
        self.lstm_model = LSTMModel(self.vocab_size)
        self.lstm_model.load_state_dict(torch.load(lstm_model_path))
        self.lstm_model.eval()
    
    def preprocess_domain(self, domain):
        domain = domain.split('.')[0].lower()
        return domain
    
    def extract_bigrams(self, domain):
        domain = re.sub(r'[^a-zA-Z0-9]', '', domain)
        return [domain[i:i+2] for i in range(len(domain)-1)]
    
    def text_to_sequence(self, text, max_len=20):
        word_to_index = {word: idx + 1 for idx, word in enumerate(self.vectorizer.get_feature_names_out())}
        seq = [word_to_index.get(word, 0) for word in text.split()]
        return seq[:max_len] + [0] * (max_len - len(seq))
    
    def detect_phishing(self, domains):
        processed_domains = [self.preprocess_domain(domain) for domain in domains]
        bigrams = [" ".join(self.extract_bigrams(domain)) for domain in processed_domains]
        X_tfidf = self.vectorizer.transform(bigrams)
        rf_preds_prob = self.rf_model.predict_proba(X_tfidf)[:, 1]
        X_seq = np.array([self.text_to_sequence(text) for text in bigrams])
        X_tensor = torch.tensor(X_seq, dtype=torch.long)
        with torch.no_grad():
            lstm_outputs = self.lstm_model(X_tensor)
            lstm_preds_prob = torch.softmax(lstm_outputs, dim=1)[:, 1].numpy()
        ensemble_preds_prob = (rf_preds_prob + lstm_preds_prob) / 2
        results = []
        for i, domain in enumerate(domains):
            results.append({
                'domain': domain,
                'rf_phishing_prob': rf_preds_prob[i],
                'lstm_phishing_prob': lstm_preds_prob[i],
                'ensemble_phishing_prob': ensemble_preds_prob[i],
                'is_phishing': ensemble_preds_prob[i] > 0.5
            })
        
        return results


rf_model_path = 'random_forest_phishing_model1.joblib'
lstm_model_path = 'lstm_phishing_model1.pth'
vectorizer_path = 'tfidf_vectorizer1.joblib'


detector = PhishingDetector(
    rf_model_path, 
    lstm_model_path, 
    vectorizer_path
)


test_domains = [
    'secure-login-bank.com',
    'verify-account-now.net',
    'paypal-confirm.org',
    'bmsit.in',
    'google.com',
    'microsoft.com',
    'github.com',
    "netflix.com",
    'goog1e.com',
    'pаypal.com'  
]


results = detector.detect_phishing(test_domains)

print("Phishing Detection Results:")
print("-" * 50)
for result in results:
    print(f"Domain: {result['domain']}")
    print(f"  Random Forest Prob: {result['rf_phishing_prob']:.4f}")
    print(f"  LSTM Prob: {result['lstm_phishing_prob']:.4f}")
    print(f"  Ensemble Prob: {result['ensemble_phishing_prob']:.4f}")
    print(f"  Likely Phishing: {'Yes' if result['is_phishing'] else 'No'}")
    print()

