from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from pathlib import Path

# Load the text data
file_path = 'lirik_lagu_test.txt'

# Initialize a ByteLevelBPETokenizer
tokenizer = Tokenizer(BPE())

# Train the tokenizer on the text file
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=[file_path], trainer=trainer)

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]