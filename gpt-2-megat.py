import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import pickle

class LyricsDataset(Dataset):
    def __init__(self, encoded_data, max_length):
        self.encoded_data = encoded_data
        self.max_length = max_length

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        data = self.encoded_data[idx]
        if len(data) > self.max_length:
            data = data[:self.max_length]
        padding_length = self.max_length - len(data)
        return torch.tensor(data + [0] * padding_length, dtype=torch.long)

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Load encoded data
with open('encoded_train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('encoded_test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)

# Parameters
max_length = 50
batch_size = 8
epochs = 3

# Prepare datasets and dataloaders
train_dataset = LyricsDataset(train_data, max_length)
test_dataset = LyricsDataset(test_data, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.resize_token_embeddings(len(tokenizer))

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader)}')

# Save the model
model.save_pretrained('fine_tuned_gpt2')
tokenizer.save_pretrained('fine_tuned_gpt2')
