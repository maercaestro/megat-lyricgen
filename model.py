import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tokenizers import Tokenizer
from tokenizers.models import BPE

# Load encoded data
with open('encoded_train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('encoded_test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Define the GPT model
class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_encoder_layers, dim_feedforward)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src):
        seq_length = src.size(1)
        src = self.embedding(src) + self.pos_encoder[:, :seq_length, :]
        src = self.transformer(src, src)
        output = self.fc_out(src)
        return output

# Model parameters
tokenizer = Tokenizer(BPE())
vocab_size = tokenizer.get_vocab_size()
d_model = 512
nhead = 8
num_encoder_layers = 6
dim_feedforward = 2048
max_seq_length = 50

model = GPTModel(vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def train(model, train_data, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_data:
            if len(data) < max_seq_length:
                continue
            src = torch.tensor(data[:max_seq_length]).unsqueeze(0)
            tgt = torch.tensor(data[1:max_seq_length+1]).unsqueeze(0)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_data)}')

train(model, train_data, criterion, optimizer, num_epochs=10)
