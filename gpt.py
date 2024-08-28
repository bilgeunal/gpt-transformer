import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class TextGeneratorModel(nn.Module):
    def __init__(self, vocab_size):
        super(TextGeneratorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 200)
        self.lstm = nn.LSTM(200, 50, 2, batch_first=True)
        self.fc = nn.Linear(50, vocab_size)
     
    def forward(self, x):
        embed_vector = self.embedding(x)
        lstm_out, _ = self.lstm(embed_vector)
        logits = self.fc(lstm_out)
        return logits

    def generate(self, start_seq, char_to_idx, idx_to_char, max_length=100):
        self.eval()
        generated_text = start_seq
        
        input_seq = torch.tensor([[char_to_idx[char] for char in start_seq]], dtype=torch.long)
        
        for _ in range(max_length - len(start_seq)):
            with torch.no_grad():
                output = self(input_seq)
                last_output = output[:, -1, :]  # Get the last output in the sequence
                probabilities = torch.softmax(last_output, dim=1)
                next_char_idx = torch.multinomial(probabilities, 1).item()
                next_char = idx_to_char[next_char_idx]
                generated_text += next_char
                
                input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_char_idx]], dtype=torch.long)], dim=1)
            
        return generated_text

def load_data(text):
    chars = sorted(set(text))
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
    vocab_size = len(chars)
    return text, char_to_idx, idx_to_char, vocab_size

def prepare_sequences(text, char_to_idx, seq_length, batch_size):
    sequences = []
    targets = []
    
    # Generate sequences and targets
    for i in range(len(text) - seq_length):
        sequences.append(text[i:i + seq_length])
        targets.append(text[i + seq_length])
    
    # Convert sequences and targets to tensor format
    X = torch.tensor([[char_to_idx[char] for char in seq] for seq in sequences], dtype=torch.long)
    y = torch.tensor([char_to_idx[target] for target in targets], dtype=torch.long)
    
    # Calculate number of batches
    num_batches = len(sequences) // batch_size
    X = X[:num_batches * batch_size]  # Trim to make sure X has exactly num_batches * batch_size entries
    y = y[:num_batches * batch_size]  # Trim to make sure y has exactly num_batches * batch_size entries
    
    # Reshape tensors to include batch dimension
    X = X.view(num_batches, batch_size, seq_length)
    y = y.view(num_batches, batch_size)
    
    return X, y





if __name__ == "__main__":
    with open('input.txt', 'r') as file:
        text = file.read()

    # Hyperparameters
    dk = 64
    num_epochs = 10  # Set the number of epochs for training
    learning_rate = 0.0001  # Adjust learning rate if necessary
    batch_size = 4
    seq_length = 7

    text, char_to_idx, idx_to_char, vocab_size = load_data(text)
    X, y = prepare_sequences(text, char_to_idx, seq_length=seq_length, batch_size=batch_size)

    model = TextGeneratorModel(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in range(X.size(0)):  # Iterate over batches
            inputs = X[batch]
            targets = y[batch]

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, vocab_size)  # Reshape outputs to match targets
            target = targets.view(-1)  # Flatten targets

            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (X.size(0))
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # Example text generation
    start_char = 'a'  # Starting character for generation
    generated_text = model.generate(start_char, char_to_idx, idx_to_char, max_length=100)
    print("Generated Text:")
    print(generated_text)



