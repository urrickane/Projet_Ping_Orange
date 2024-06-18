from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pickle
import spacy

# Définir le modèle LSTMClassifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_rate=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # Adjusted for bidirectional LSTM

    def forward(self, text, lengths):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)  # Concatenate the hidden states from both directions
        output = self.fc(hidden)
        return output

# Charger le vocabulaire et le modèle
with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab), 100, 256, 3).to(device)
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
print("Le modèle a été chargé.")

# Charger le modèle de tokenisation français de spacy
nlp = spacy.load("fr_core_news_sm")

# Définir l'application Flask
app = Flask(__name__)
CORS(app)  # Activer CORS pour toutes les routes

def tokenizer(text):
    return [token.text for token in nlp(text)]

def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)]

@app.route('/predict', methods=['POST'])

def predict():
    data = request.get_json(force=True)
    sentence = data['sentence']
    processed_text = torch.tensor(text_pipeline(sentence), dtype=torch.int64).unsqueeze(0).to(device)
    lengths = torch.tensor([processed_text.size(1)], dtype=torch.int64).to(device)
    output = model(processed_text, lengths)
    prediction = torch.argmax(output, dim=1).item()
    sentiment = {0: 'Achat', 1: 'Rendez-vous', 2: 'Autres'}
    return jsonify({'category': sentiment[prediction]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
