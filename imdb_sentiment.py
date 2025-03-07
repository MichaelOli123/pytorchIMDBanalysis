import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
import spacy

spacy_en = spacy.load('en_core_web_sm')

TEXT = Field(sequential=True, tokenize='spacy', tokenizer_language='en_core_web_sm', include_lengths=True)
LABEL = Field(sequential=False, use_vocab=False, is_target=True)

train_data, test_data = IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size=25000, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)

BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_within_batch=True,
    sort_key=lambda x: len(x.text)
)

class SentimentAnalysisModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        return output

input_dim = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1  
dropout = 0.5

model = SentimentAnalysisModel(input_dim, embedding_dim, hidden_dim, output_dim, dropout)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)
def train(model, iterator, optimizer, criterion):
    model.train()
    
    epoch_loss = 0
    epoch_accuracy = 0
    
    for batch in iterator:
        text, text_lengths = batch.text
        label = batch.label
        
        optimizer.zero_grad()
        
        predictions = model(text, text_lengths).squeeze(1)
        
        loss = criterion(predictions, label)
        acc = binary_accuracy(predictions, label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_accuracy += acc.item()
    
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()
    
    epoch_loss = 0
    epoch_accuracy = 0
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            label = batch.label
            
            predictions = model(text, text_lengths).squeeze(1)
            
            loss = criterion(predictions, label)
            acc = binary_accuracy(predictions, label)
            
            epoch_loss += loss.item()
            epoch_accuracy += acc.item()
    
    return epoch_loss / len(iterator), epoch_accuracy / len(iterator)


N_EPOCHS = 10

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    
    print(f'Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Train Accuracy: {train_acc*100:.2f}%')
    print(f'Epoch {epoch+1} | Test Loss: {test_loss:.3f} | Test Accuracy: {test_acc*100:.2f}%')
