import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import joblib

# Load phishing dataset from CSV
df = pd.read_csv("phishing_url_dataset.csv")

# Convert 'type' column to numerical labels: phishing = 1, benign = 0
df["label"] = df["type"].apply(lambda x: 1 if x == "phishing" else 0)

# Convert URLs into bigram features
def extract_bigrams(url):
    url = re.sub(r'[^a-zA-Z0-9]', '', url.lower())  # Remove special characters
    return ["".join(url[i:i+2]) for i in range(len(url)-1)]

df["bigrams"] = df["url"].apply(lambda x: " ".join(extract_bigrams(x)))

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df["bigrams"], df["label"], test_size=0.2, random_state=42)

# TF-IDF for Random Forest
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Convert URL bigrams into indexed sequences for LSTM
word_to_index = {word: idx + 1 for idx, word in enumerate(vectorizer.get_feature_names_out())}

def text_to_sequence(text, max_len=20):
    seq = [word_to_index.get(word, 0) for word in text.split()]
    return seq[:max_len] + [0] * (max_len - len(seq))

X_train_seq = np.array([text_to_sequence(text) for text in X_train])
X_test_seq = np.array([text_to_sequence(text) for text in X_test])

# Define LSTM Dataset
class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Dataloader
batch_size = 2
train_data = DataLoader(URLDataset(X_train_seq, y_train), batch_size=batch_size, shuffle=True)
test_data = DataLoader(URLDataset(X_test_seq, y_test), batch_size=batch_size, shuffle=False)

# Define LSTM Model
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

# Initialize LSTM Model
vocab_size = len(word_to_index) + 1
lstm_model = LSTMModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.01)

# Train LSTM
epochs = 5
for epoch in range(epochs):
    for X_batch, y_batch in train_data:
        optimizer.zero_grad()
        output = lstm_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

# Get Predictions from RF
rf_preds = rf_model.predict_proba(X_test_tfidf)[:, 1]

# Get Predictions from LSTM
lstm_preds = []
with torch.no_grad():
    for X_batch, _ in test_data:
        outputs = lstm_model(X_batch)
        lstm_preds.extend(torch.softmax(outputs, dim=1)[:, 1].numpy())

# Ensemble: Averaging RF & LSTM Predictions
final_preds = (np.array(rf_preds) + np.array(lstm_preds)) / 2
final_labels = [1 if p > 0.5 else 0 for p in final_preds]

# Print Results
print("Final Predictions:", final_labels)

# Save the models
joblib.dump(rf_model, "random_forest_model.joblib")
torch.save(lstm_model.state_dict(), "lstm_model.pth")
# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

