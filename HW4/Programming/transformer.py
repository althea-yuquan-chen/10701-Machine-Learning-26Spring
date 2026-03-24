import pickle
import tqdm
import torch
import random
import numpy as np
from torch import nn, optim

from reference_data import get_dataloader
from reference_data_q4 import get_dataloader_q4

torch.manual_seed(10701)
random.seed(10701)
np.random.seed(10701)

# Set device based on GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### DO NOT CHANGE ###
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super().__init__()
        self.table_shape = (vocab_size, embed_dim)
        self.table = torch.randn(self.table_shape, requires_grad=True)

    def forward(self, word_indices):
        return self.table[word_indices]

### DO NOT CHANGE ###
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        # Same size as input matrix (for element-wise addition with input matrix)
        self.encoding = torch.zeros(max_len, d_model, requires_grad=False).to(device)

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # Compute positional encoding to incorporate the positional information of words

    def forward(self, x):
        seq_len = x.shape[1]
        return self.encoding[:seq_len, :]

### DO NOT CHANGE ###
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=500, drop_prob=0.1):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class AttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # TODO: Initialize, query (W_q), key (W_k), value (W_v) weight matrices and softmax layer
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

    def forward(self, inp_q, inp_k, inp_v, mask=None):
        # TODO: Implement forward 
        Q = self.W_q(inp_q)
        K = self.W_k(inp_k)
        V = self.W_v(inp_v)

        # Calculate attention scores using scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super().__init__()
        # TODO: Initialize n_heads of AttentionLayer and linear layer
        # Hint: nn.ModuleList()
        self.n_heads = n_heads
        self.attention_heads = nn.ModuleList([AttentionLayer(d_model, d_k, d_v) for _ in range(n_heads)])

        self.w_o = nn.Linear(n_heads * d_v, d_model)

        
    def forward(self, inp_q, inp_k, inp_v, mask=None):
        # TODO: Implement forward 
        head_outputs = []
        for attention_head in self.attention_heads:
            head_output = attention_head(inp_q, inp_k, inp_v, mask)
            head_outputs.append(head_output)

        # Concatenate the outputs from all attention heads
        concatenated = torch.cat(head_outputs, dim=-1)
        return self.w_o(concatenated)


class Residual(nn.Module):
    def __init__(self, module, d_model, drop_p=0.1):
        super().__init__()
        self.module = module

        # TODO: Initialize layer normalization and dropout
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, *inp):
        # TODO: Implement forward
        # Hint: Is the residual connection before or after dropout?
        # sublayer_out = self.module(*inp)
        # return self.layer_norm(inp[0] + sublayer_out)
        normed_inp = [self.layer_norm(x) for x in inp]
        sublayer_out = self.module(*normed_inp)
        return inp[0] + sublayer_out
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_lin):
        super().__init__()

        # TODO: Initialize feed forward network with ReLU activation
        self.linear1 = nn.Linear(d_model, d_lin)
        self.linear2 = nn.Linear(d_lin, d_model)
        #self.relu = nn.ReLU()
        self.gelu = nn.GELU()

    def forward(self, inp):
        # TODO: Implement forward
        x = self.gelu(self.linear1(inp))
        x = self.linear2(x)
        return x
    

# class FeedForward(nn.Module):
#     def __init__(self, d_model, d_lin):
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, d_lin)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(d_lin, d_model)

#     def forward(self, x):
#         return self.fc2(self.relu(self.fc1(x)))

# class Residual(nn.Module):
#     def __init__(self, sublayer, d_model, drop_p=0.1):
#         super().__init__()
#         self.sublayer = sublayer
#         self.layer_norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(drop_p)

#     def forward(self, *inp):
#         sublayer_out = self.sublayer(*inp)
#         sublayer_out = self.dropout(sublayer_out)
#         return self.layer_norm(inp[0] + sublayer_out)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_lin):
        super().__init__()
        # TODO: Initialize MultiHeadAttention and FeedForward modules with Residual layers for both
        mha = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.res_attn = Residual(mha, d_model)
        ffn = FeedForward(d_model, d_lin)
        self.res_ffn = Residual(ffn, d_model)

    def forward(self, inp):
        # TODO: Implement forward
        x = self.res_attn(inp, inp, inp)
        x = self.res_ffn(x)
        return x

class Encoder(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_lin, n_layers, vocab_size):
        super().__init__()
        # TODO: Initialize TransformerEmbedding and n_layers of EncoderLayer
        # Hint: nn.ModuleList()
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_k, d_v, d_lin) for _ in range(n_layers)])

    def forward(self, inp):
        # TODO: Implement forward by embedding the input and passing it through all layers
        for layer in self.layers:
            inp = layer(inp)
        
        return inp


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_classes,
        n_heads,
        d_model,
        d_k,
        d_v,
        d_lin,
        n_layers,
    ):
        super().__init__()
        # TODO: Initialize Encoder and Classifier of Transformer model
        self.encoder = Encoder(n_heads, d_model, d_k, d_v, d_lin, n_layers, vocab_size)
        self.classifier = nn.Linear(d_model, n_classes)
        self.embedding = TransformerEmbedding(vocab_size, d_model)

    def forward(self, inp):
        # TODO: Implement forward
        # Hint: Return the probability that the input sequence expresses positive sentiment
        # Hint: What should the input to the classifier be?
        embedded = self.embedding(inp)
        encoded = self.encoder(embedded)
        # Take the output of the first token (CLS token) for classification
        cls_output = encoded[:, 0, :]
        return self.classifier(cls_output)


# Feel free to reuse your RNN training loop!
def train(model, train_loader, epoch, optimizer, criterion):
    train_avg_loss = 0
    num_correct = 0

    # TODO: Set model to training mode

    batch_bar = tqdm.tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Train epoch {epoch}")
    for text, label in train_loader:
        # TODO: Move data to device
        texts = text.to(device)
        labels = label.to(device)

        current_batch_size = texts.size(0)
        optimizer.zero_grad()

        # TODO: Get model prediction and calculate loss
        predictions = model(texts)
        loss = criterion(predictions, labels)

        # TODO: Calculate number of correct predictions
        train_avg_loss += loss.item() * current_batch_size
        predicted = torch.argmax(predictions, dim=-1)
        num_correct += (predicted == labels).sum().item()

        # TODO: Backward pass, update weights, and zero gradients
        loss.backward()
        optimizer.step()

        batch_bar.set_postfix(loss="{:.06f}".format(loss))
        batch_bar.update()
        
        
    train_avg_loss /= len(train_loader.dataset)
    train_accuracy = num_correct / len(train_loader.dataset)
    return train_avg_loss, train_accuracy


def test(model, test_loader, epoch, criterion):
    test_avg_loss = 0
    num_correct = 0

    # TODO: Set model to eval mode
    model.eval()

    batch_bar = tqdm.tqdm(total=len(test_loader), dynamic_ncols=True, leave=False, position=0, desc=f"Validate epoch {epoch}")

    with torch.no_grad():
        for text, label in test_loader:
            # TODO: Move data to device
            texts = text.to(device)
            labels = label.to(device)

            # TODO: Get model prediction and calculate loss
            predictions = model(texts)
            loss = criterion(predictions, labels)

            # TODO: Calculate number of correct predictions
            test_avg_loss += loss.item() * texts.size(0)
            predicted = torch.argmax(predictions, dim=-1)
            num_correct += (predicted == labels).sum().item()


            batch_bar.set_postfix(loss="{:.06f}".format(loss))
            batch_bar.update()

    test_avg_loss /= len(test_loader.dataset)
    test_accuracy = num_correct / len(test_loader.dataset)
    return test_avg_loss, test_accuracy


def test_autograder(model, test_loader):
    model.eval()
    predicted_labels_list = []
    with torch.no_grad():
        # Iterate through the test_loader to get the text and labels
        for text, _ in test_loader:
            text = text.to(device)
            prediction = model(text)
            binary_prediction = prediction.argmax(dim=1)
            #Add (text, prediction) pairs to the list
            for pred in binary_prediction:
                predicted_labels_list.append(pred.item())
        
    # Save the collected data into a pickle file
    with open("predicted_labels.pkl", 'wb') as f:
        pickle.dump(predicted_labels_list, f)


def run(num_epochs, model, train_loader, test_loader, optimizer, criterion):
    train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = [], [], [], []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        # TODO: Train the model
        train_loss, train_acc = train(model, train_loader, epoch, optimizer, criterion)
        print(f"Train loss: {train_loss:.06f} | Accuracy: {train_acc*100:.04f}%")

        # TODO: Test the model
        test_loss, test_acc = test(model, test_loader, epoch, criterion)
        print(f"Test loss: {test_loss:.06f} | Accuracy: {test_acc*100:.04f}%")
    
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
    return train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist

import matplotlib.pyplot as plt
def plot_loss(train_loss_hist, test_loss_hist, save_name="transformer_loss.png"):
    epochs = range(len(train_loss_hist))
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_hist, label='Train Loss', marker='o', color='blue')
    plt.plot(epochs, test_loss_hist, label='Test Loss', marker='s', color='orange')
    
    plt.title('Transformer: Training and Testing Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    

    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n Loss plot successfully saved to {save_name}!")



def main(
    #Hyperparameters
    q_44 = False,
    vocabulary_size = 20000,
    batch_size = 128,
    max_review_length = 50,
    lr = 1e-3,
    num_epochs = 10,
    n_classes = 2,
    n_heads = 3,
    d_model = 64,
    d_k = 32,
    d_v = 32,
    d_lin = 64,
    n_layers = 3
):

    # TODO: Initialize Transformer model, criterion and optimizer
    model = Transformer(vocab_size=vocabulary_size, n_classes=n_classes, n_heads=n_heads, d_model=d_model, d_k=d_k, d_v=d_v, d_lin=d_lin, n_layers=n_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Calculate total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")

    if q_44:
        train_dataloader, test_dataloader, autograder_dataloader, vocabulary = get_dataloader_q4(vocabulary_size, max_review_length, batch_size)

    else:
        train_dataloader, test_dataloader, vocab = get_dataloader(vocabulary_size, max_review_length, batch_size)

    # TODO: Train and test the model
    train_loss_list, train_acc_list, test_loss_list, test_acc_list = run(num_epochs, model, train_dataloader, test_dataloader, optimizer, criterion)
    plot_loss(train_loss_list, test_loss_list, save_name="transformer_loss_plot.png")

    if q_44:
        # Submit "predicted_labels.pkl" to Gradescope Homework 4 Programming (4.4 predicted_labels.pkl) 
        test_autograder(model, autograder_dataloader)

    return train_loss_list, test_loss_list


if __name__ == '__main__':
    main(q_44=True)
    
