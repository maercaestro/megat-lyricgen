import random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
import pickle
import json

# Paths
file_path = 'lirik_lagu_test.txt'       # Change this to the correct file path
train_file_path = 'lirik_train.txt'
test_file_path = 'lirik_test.txt'
tokenizer_path = 'lyrics_tokenizer'

# Step 1: Split the Text File
def split_text_file(file_path, train_file_path, test_file_path, split_ratio=0.8):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    random.shuffle(lines)
    train_size = int(split_ratio * len(lines))
    train_lines = lines[:train_size]
    test_lines = lines[train_size:]
    
    with open(train_file_path, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_lines)
    
    with open(test_file_path, 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_lines)

split_text_file(file_path, train_file_path, test_file_path)

# Step 2: Train the Tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=[train_file_path, test_file_path], trainer=trainer)

# Define post-processing to add special tokens
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

# Save the tokenizer
tokenizer.save(f"{tokenizer_path}.json")

# Step 3: Encode the Data
def encode_file(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    encoded_lines = [tokenizer.encode(line.strip()).ids for line in lines]
    return encoded_lines

tokenizer = Tokenizer.from_file(f"{tokenizer_path}.json")
train_data = encode_file(train_file_path, tokenizer)
test_data = encode_file(test_file_path, tokenizer)

# Step 4: Save the Encoded Data
with open('encoded_train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

with open('encoded_test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

print("Data splitting, tokenization, and encoding completed successfully.")
