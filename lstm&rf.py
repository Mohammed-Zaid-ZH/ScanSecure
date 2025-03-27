import numpy as np
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
import joblib
df = pd.read_csv('phishing_detection_dataset.csv')
def preprocess_domain(domain):
    domain = domain.split('.')[0].lower()
    return domain
df['processed_domain'] = df['domain'].apply(preprocess_domain)
def extract_bigrams(domain):
    domain = re.sub(r'[^a-zA-Z0-9]', '', domain)
    return [domain[i:i+2] for i in range(len(domain)-1)]

df['bigrams'] = df['processed_domain'].apply(lambda x: " ".join(extract_bigrams(x)))
X = df['bigrams']
y = df['is_phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_preds = rf_model.predict_proba(X_test_tfidf)[:, 1]
word_to_index = {word: idx + 1 for idx, word in enumerate(vectorizer.get_feature_names_out())}

def text_to_sequence(text, max_len=20):
    seq = [word_to_index.get(word, 0) for word in text.split()]
    return seq[:max_len] + [0] * (max_len - len(seq))
X_train_seq = np.array([text_to_sequence(text) for text in X_train])
X_test_seq = np.array([text_to_sequence(text) for text in X_test])


class URLDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
train_data = DataLoader(URLDataset(X_train_seq, y_train), batch_size=batch_size, shuffle=True)
test_data = DataLoader(URLDataset(X_test_seq, y_test), batch_size=batch_size, shuffle=False)

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
vocab_size = len(word_to_index) + 1
lstm_model = LSTMModel(vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
epochs = 10
for epoch in range(epochs):
    lstm_model.train()
    total_loss = 0
    for X_batch, y_batch in train_data:
        optimizer.zero_grad()
        output = lstm_model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data)}')

lstm_model.eval()
lstm_preds = []
with torch.no_grad():
    for X_batch, _ in test_data:
        outputs = lstm_model(X_batch)
        lstm_preds.extend(torch.softmax(outputs, dim=1)[:, 1].numpy())
final_preds = (np.array(rf_preds) + np.array(lstm_preds)) / 2
final_labels = [1 if p > 0.5 else 0 for p in final_preds]
print("\nRandom Forest Performance:")
rf_test_preds = rf_model.predict(X_test_tfidf)
print(classification_report(y_test, rf_test_preds))
lstm_test_labels = [1 if p > 0.5 else 0 for p in lstm_preds]
print("\nLSTM Performance:")
print(classification_report(y_test, lstm_test_labels))

print("\nEnsemble Performance:")
print(classification_report(y_test, final_labels))
joblib.dump(rf_model, "random_forest_phishing_model1.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer1.joblib")
torch.save(lstm_model.state_dict(), "lstm_phishing_model1.pth")

print("\nModels and vectorizer saved successfully!")
