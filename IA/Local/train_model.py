import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import spacy

# Charger le modèle de tokenisation français de spacy
nlp = spacy.load("fr_core_news_sm")

# CustomTextDataset class for handling texts and labels directly
class CustomTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

# Load and inspect the dataset
df = pd.read_csv('datav3.csv')
print("First few rows of the dataset:")
print(df.head())

# Check the class distribution
class_counts = df.iloc[:, 1].value_counts()
print("Class distribution:")
print(class_counts)

# Verify that each class has enough instances for stratified splitting
if class_counts.min() < 2:
    raise ValueError("Each class must have at least 2 instances for stratified splitting.")

# Load and split data according to 70-15-15 model
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    df.iloc[:, 0].tolist(),
    df.iloc[:, 1].tolist(),
    test_size=0.3,
    stratify=df.iloc[:, 1].tolist()
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    stratify=temp_labels
)

# Check the class distribution in each split
train_class_counts = pd.Series(train_labels).value_counts()
val_class_counts = pd.Series(val_labels).value_counts()
test_class_counts = pd.Series(test_labels).value_counts()

print("Training set class distribution:")
print(train_class_counts)
print("Validation set class distribution:")
print(val_class_counts)
print("Test set class distribution:")
print(test_class_counts)

# Create the datasets
train_dataset = CustomTextDataset(train_texts, train_labels)
val_dataset = CustomTextDataset(val_texts, val_labels)
test_dataset = CustomTextDataset(test_texts, test_labels)

# Define tokenizer and build vocab
def tokenizer(text):
    return [token.text for token in nlp(text)]

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_dataset.texts), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

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

# Pipeline and collate function adjustments
def text_pipeline(x):
    return [vocab[token] for token in tokenizer(x)]

def label_pipeline(x):
    label_mapping = {'Achat': 0, 'Rendez-vous': 1, 'Autres': 2}
    return label_mapping[x]

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))  # This should return a class index, not a vector
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        if processed_text.size(0) == 0:
            processed_text = torch.tensor([vocab["<unk>"]], dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)  # Ensure this is a 1D tensor of class indices
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths, dtype=torch.int64)

    return text_list, label_list, lengths

# Setup data loaders with larger batch size
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)

# Model setup and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(len(vocab), 100, 256, 3).to(device)  # Adjusted for 3 classes
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.01)
criterion = nn.CrossEntropyLoss().to(device)

def train(dataloader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for text, label, lengths in dataloader:
        optimizer.zero_grad()
        text, lengths, label = text.to(device), lengths.to(device), label.to(device)
        output = model(text, lengths)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        correct += (predicted == label).sum().item()
        total += label.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

def validate(dataloader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for text, label, lengths in dataloader:
            text, lengths, label = text.to(device), lengths.to(device), label.to(device)
            output = model(text, lengths)
            loss = criterion(output, label)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

num_epochs = 100

# Lists for storing loss and accuracy
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_loss = float('inf')
early_stopping_patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    train_loss, train_accuracy = train(train_dataloader)
    val_loss, val_accuracy = validate(val_dataloader)

    # Store loss and accuracy
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')  # Save the best model
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

# Plot loss and accuracy curves
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.show()

plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Function to test the model with a new input sentence
def predict_sentiment(sentence):
    model.load_state_dict(torch.load('best_model.pt'))  # Load the best model
    model.eval()
    with torch.no_grad():
        processed_text = torch.tensor(text_pipeline(sentence), dtype=torch.int64).unsqueeze(0).to(device)
        lengths = torch.tensor([processed_text.size(1)], dtype=torch.int64).to(device)
        output = model(processed_text, lengths)
        prediction = torch.argmax(output, dim=1).item()
        sentiment = {0: 'Achat', 1: 'Rendez-vous', 2: 'Autres'}
        return sentiment[prediction]

# Evaluate model on test set
def evaluate_model(dataloader):
    model.load_state_dict(torch.load('best_model.pt'))  # Load the best model
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for text, label, lengths in dataloader:
            text, lengths, label = text.to(device), lengths.to(device), label.to(device)
            output = model(text, lengths)
            predictions = torch.argmax(output, dim=1)
            all_labels.extend(label.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    print(classification_report(all_labels, all_predictions, target_names=['Achat', 'Rendez-vous', 'Autres']))

evaluate_model(test_dataloader)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
