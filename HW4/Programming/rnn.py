import math
import tqdm
import string
import time
import random
import numpy as np
import torch
from torch import nn, optim

from reference_data import get_dataloader

# Set device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, output_size, batch_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        # TODO: Initialize embeddings, model layers, layer normalization, and activation function
        self.embedding = nn.Embedding(vocab_size, input_size, padding_idx=0)
        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.tanh = nn.Tanh()


    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        # TODO: Implement forward pass including layer normalization
        embedded = self.embedding(input)

        outputs = []
        seq_len = embedded.size(1)

        for t in range(seq_len):
            # Fetch the input embedding for the current time step i
            x_i = embedded[:, t, :] # shape: (B, input_size)
            
            # Mathematical formula from instructions: h_i = tanh(LayerNorm(W_hh * h_{i-1} + W_xh * x_i))
            hidden = self.tanh(self.layer_norm(self.h2h(hidden) + self.i2h(x_i)))
            
            # Calculate output logits (Note: PyTorch's CrossEntropyLoss expects raw logits, not softmax probabilities)
            output = self.h2o(hidden) # shape: (B, output_size)
            outputs.append(output)
            
        # Stack all outputs along the sequence dimension (dim=1)
        # Returns shape: (B, L, output_size)
        return torch.stack(outputs, dim=1)
    
    def init_hidden(self) -> torch.Tensor:
        # TODO: Initialize hidden state to zeros
        return torch.zeros(self.batch_size, self.hidden_size).to(device)


def train(data_loader, model, criterion, optimizer):
    test_avg_loss = 0
    num_correct = 0

    # TODO: Set model to training mode
    model.train()

    # TODO: Implement training loop
    # Hint: See reference_data.py for the format of data_loader batches
    for texts, labels in tqdm.tqdm(data_loader, leave=False):

        # TODO: Convert labels to LongTensors and move to device
        texts = texts.to(device)
        labels = labels.to(device)

        current_batch_size = texts.size(0)

        # TODO: Initialize hidden state
        hidden = model.init_hidden()[:current_batch_size, :]
        optimizer.zero_grad()

        # TODO: Perform forward pass and calculate avg loss over all time steps
        outputs = model(texts, hidden)
        loss = 0
        seq_len = outputs.size(1)
        for t in range(seq_len):
            # criterion computes the average loss across the batch for time step t
            loss += criterion(outputs[:, t, :], labels)
        
        loss = loss / seq_len  # Average loss over all time steps

        # TODO: Calculate number of correct predictions
        test_avg_loss += loss.item() * current_batch_size
        last_output = outputs[:, -1, :] # shape: (B, num_classes)
        
        predicted = torch.argmax(last_output, dim=-1)
        num_correct += (predicted == labels).sum().item()

        # TODO: Backward pass, update weights, and zero gradients
        loss.backward()
        optimizer.step()


    # Uncomment the line below once the training loop is implemented to print 
    # the loss and accuracy at the end of each epoch
    print(f"Train loss: {test_avg_loss/len(data_loader.dataset)} | Accuracy: {num_correct/len(data_loader.dataset)}")
    return test_avg_loss/len(data_loader.dataset), num_correct/len(data_loader.dataset)


def test(data_loader, rnn, criterion):
    test_avg_loss = 0
    num_correct = 0

    # TODO: Set model to eval mode
    rnn.eval()

    # TODO: Implement testing loop
    # Hint: Do we want to update gradients when testing?
    with torch.no_grad():
        for texts, labels in tqdm.tqdm(data_loader, leave=False):
            texts = texts.to(device)
            labels = labels.to(device)
            current_batch_size = texts.size(0)
            hidden = rnn.init_hidden()[:current_batch_size, :]
            outputs = rnn(texts, hidden)
            loss = 0
            seq_len = outputs.size(1)
            for t in range(seq_len):
                loss += criterion(outputs[:, t, :], labels)
            loss = loss / seq_len  # Average loss over all time steps
            test_avg_loss += loss.item() * current_batch_size

            last_output = outputs[:, -1, :]
            predicted = torch.argmax(last_output, dim=-1)
            num_correct += (predicted == labels).sum().item()

        # Uncomment the line below once the training loop is implemented to print 
        # the loss and accuracy at the end of each epoch
    print(f"Test loss: {test_avg_loss/len(data_loader.dataset)} | Accuracy: {num_correct/len(data_loader.dataset)}")
    return test_avg_loss/len(data_loader.dataset), num_correct/len(data_loader.dataset)


def run(num_epochs, train_dataloader, test_dataloader, rnn, criterion, optimizer):
    train_loss_list = []
    test_loss_list = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        # TODO: Perform one epoch of training and testing
        train_loss, _ = train(train_dataloader, rnn, criterion, optimizer)
        test_loss, _ = test(test_dataloader, rnn, criterion)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

    return train_loss_list, test_loss_list



def main(
    # Hyperparameters
    vocabulary_size = 20000,
    batch_size = 128,
    input_size = 64,
    hidden_size = 128,
    max_review_length = 50,
    lr = 1e-4,
    num_epochs = 10,
    num_classes = 2
):

    train_dataloader, test_dataloader, full_vocab = get_dataloader(vocabulary_size, max_review_length, batch_size)

    # TODO: Initialize model
    model = RNN(vocab_size=vocabulary_size, input_size=input_size, hidden_size=hidden_size, output_size=num_classes, batch_size=batch_size).to(device)

    # TODO: Initialize loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # TODO: Run training and testing
    train_loss_list, test_loss_list = run(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer)

    return train_loss_list, test_loss_list

if __name__ == '__main__':
    main()
    