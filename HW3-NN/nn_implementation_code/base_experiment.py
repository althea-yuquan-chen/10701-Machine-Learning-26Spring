from collections import Counter
from xml.parsers.expat import model

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from custom_modules import Linear, CrossEntropyLoss, Sigmoid

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


class FashionMNISTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize linear and sigmoid layers
        self.lin1 = Linear(28*28, 256)
        self.lin2 = Linear(256, 10)
        self.sigmoid = Sigmoid()


    def forward(self, x):
        # TODO: Implement forward pass 
        x = x.view(x.size(0), -1)  # Flatten the input
        a = self.lin1(x) # Linear layer 1
        z = self.sigmoid(a) # Sigmoid activation
        logits = self.lin2(z) # Linear layer 2
        return a, z, logits


@torch.no_grad()
def evaluate(model, loader, loss_func, device):
    """
    Evaluate the model over the full dataset.
    Returns (loss, accuracy) averaged over total samples, not batches.
    """
    # TODO: Implement evaluation function
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        a, z, logits = model(x)
        loss = loss_func(logits, y)
        total_loss += loss.item() * x.size(0)  # Accumulate total loss
        _, predicted = torch.max(logits, 1)  # Get predicted class
        total_correct += (predicted == y).sum().item()  # Count correct predictions
        total_samples += x.size(0)  # Count total samples

    average_loss = total_loss / total_samples if total_samples > 0 else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0

    return average_loss, accuracy


def train(model, loader, optimizer, loss_func, device):
    """Train the model for one epoch. Mutates model in place, returns nothing."""
    # TODO: Implement training loop for one epoch
    model.train()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # Clear gradients
        a, z, logits = model(x)  # Forward pass
        loss = loss_func(logits, y)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters



if __name__ == "__main__":
    trainset = torchvision.datasets.FashionMNIST(root='./', train=True,
                                                 download=True, transform=transforms.ToTensor())

    testset = torchvision.datasets.FashionMNIST(root='./', train=False,
                                                download=True, transform=transforms.ToTensor())

    weights = torch.load("weights.pt")
    print(weights)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    loader_train = DataLoader(trainset, batch_size=1, shuffle=False)
    loader_test = DataLoader(testset, batch_size=1, shuffle=False)

    model = FashionMNISTModel().to(device)
    model.load_state_dict(weights)

    loss_func = CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Q1, Q2, Q3
    model.eval()
    with torch.no_grad():
        x1, y1  = next(iter(loader_train))
        x1, y1 = x1.to(device), y1.to(device)
        a1, z1, logits1 = model(x1)

        Q1 = a1[0, 10].item()
        Q2 = z1[0, 10].item()
        Q3 = torch.argmax(logits1, dim=1).item()

        print(f"Q1 (a_10): {Q1:.4f}")
        print(f"Q2 (z_20): {Q2:.4f}")
        print(f"Q3 (predicted class): {Q3}")

    # Q4, Q5, Q6
    Q4 = []
    Q5 = []
    Q6 = []

    for epoch in tqdm(range(15)):
        train(model, loader_train, optimizer, loss_func, device)
        loss, acc = evaluate(model, loader_test, loss_func, device)
        Q5.append(round(loss, 4))
        Q6.append(round(acc, 4))

        if epoch == 2:
            Q4 = model.lin2.bias.detach().cpu().numpy().tolist()
            Q4 = [round(b, 4) for b in Q4]

    print(f"\nQ4 (beta_bias at epoch 3): {Q4}")
    print(f"Q5 (Test Loss history): {Q5}")
    print(f"Q6 (Test Acc history): {Q6}")


    # Q7, Q8, Q9
    print("\n--- Starting Q7: Batch Size 5 for 50 Epochs ---")
    
    loader_train_b5 = DataLoader(trainset, batch_size=5, shuffle=False)
    loader_test_b5 = DataLoader(testset, batch_size=5, shuffle=False)
    
    model_b5 = FashionMNISTModel().to(device)
    model_b5.load_state_dict(torch.load("weights.pt"))
    optimizer_b5 = torch.optim.SGD(model_b5.parameters(), lr=0.01)
    

    for epoch in tqdm(range(50)):
        train(model_b5, loader_train_b5, optimizer_b5, loss_func, device)
        

    final_train_loss, _ = evaluate(model_b5, loader_train_b5, loss_func, device)
    _, final_test_acc = evaluate(model_b5, loader_test_b5, loss_func, device)
    
    print(f"\nQ7 - Final Training Loss: {final_train_loss:.4f}")
    print(f"Q7 - Final Test Accuracy: {final_test_acc:.4f}")

    print("\n--- Generating Q8: Confusion Matrices ---")
    def get_all_preds(model, loader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                _, _, logits = model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.numpy())
        return np.array(all_labels), np.array(all_preds)

    train_labels, train_preds = get_all_preds(model_b5, loader_train_b5)
    cm_train = confusion_matrix(train_labels, train_preds)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train)
    disp_train.plot(cmap='Blues')
    plt.title("Q8: Training Confusion Matrix")
    plt.show()

    test_labels, test_preds = get_all_preds(model_b5, loader_test_b5)
    cm_test = confusion_matrix(test_labels, test_preds)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp_test.plot(cmap='Oranges')
    plt.title("Q8: Test Confusion Matrix")
    plt.show()

    print("\n--- Generating Q9: First Misclassified Examples ---")
    
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    misclassified_found = {i: False for i in range(10)}
    count_found = 0
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    model_b5.eval()
    with torch.no_grad():
        for x, y in loader_test_b5:
            if count_found == 10:
                break
            
            x_device = x.to(device)
            _, _, logits = model_b5(x_device)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_numpy = y.numpy()
            
            for i in range(len(y_numpy)):
                true_label = y_numpy[i]
                pred_label = preds[i]
                
                if true_label != pred_label and not misclassified_found[true_label]:
                    misclassified_found[true_label] = True
                    count_found += 1
                    
                    ax = axes[true_label]
                    img = x[i].squeeze().numpy()
                    ax.imshow(img, cmap='gray')
                    ax.set_title(f"True: {classes[true_label]}\nPred: {classes[pred_label]}")
                    ax.axis('off')
                    
    plt.tight_layout()
    plt.show()

    # Q10: Batch Size Experiments
    print("\n--- Starting Q10: Batch Size Experiments ---")
    batch_sizes = [10, 50, 100]
    epochs_q10 = 50
    
    train_losses_dict = {}
    test_losses_dict = {}

    for bs in batch_sizes:
        print(f"\nTraining with Batch Size: {bs}")
        loader_train_bs = DataLoader(trainset, batch_size=bs, shuffle=False)
        loader_test_bs = DataLoader(testset, batch_size=bs, shuffle=False)
        
        model_bs = FashionMNISTModel().to(device)
        model_bs.load_state_dict(torch.load("weights.pt"))
        optimizer_bs = torch.optim.SGD(model_bs.parameters(), lr=0.01)
        
        train_losses = []
        test_losses = []
        
        for epoch in tqdm(range(epochs_q10), desc=f"BS={bs}"):
            train(model_bs, loader_train_bs, optimizer_bs, loss_func, device)
            
            # Record train loss and test loss for the current epoch
            t_loss, _ = evaluate(model_bs, loader_train_bs, loss_func, device)
            v_loss, _ = evaluate(model_bs, loader_test_bs, loss_func, device)
            
            train_losses.append(t_loss)
            test_losses.append(v_loss)
            
        train_losses_dict[bs] = train_losses
        test_losses_dict[bs] = test_losses

    # Plot Training Loss for Q10
    plt.figure(figsize=(10, 5))
    for bs in batch_sizes:
        plt.plot(range(1, epochs_q10 + 1), train_losses_dict[bs], label=f'Batch Size {bs}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Q10: Training Loss vs. Epoch for Different Batch Sizes')
    plt.legend()
    plt.show()

    # Plot Test Loss for Q10
    plt.figure(figsize=(10, 5))
    for bs in batch_sizes:
        plt.plot(range(1, epochs_q10 + 1), test_losses_dict[bs], label=f'Batch Size {bs}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Q10: Test Loss vs. Epoch for Different Batch Sizes')
    plt.legend()
    plt.show()

    # Q13: Custom Hyperparameter Experiment (Learning Rate)
    print("\n--- Starting Q13: Learning Rate Experiments ---")
    # We use control variable method: fix Batch Size = 50, vary Learning Rate
    fixed_bs = 50
    learning_rates = [0.1, 0.01, 0.001]
    
    loader_train_lr = DataLoader(trainset, batch_size=fixed_bs, shuffle=False)
    loader_test_lr = DataLoader(testset, batch_size=fixed_bs, shuffle=False)
    
    train_losses_lr = {}
    test_losses_lr = {}

    for lr in learning_rates:
        print(f"\nTraining with Learning Rate: {lr}")
        model_lr = FashionMNISTModel().to(device)
        model_lr.load_state_dict(torch.load("weights.pt"))
        optimizer_lr = torch.optim.SGD(model_lr.parameters(), lr=lr)
        
        t_losses = []
        v_losses = []
        
        for epoch in tqdm(range(epochs_q10), desc=f"LR={lr}"):
            train(model_lr, loader_train_lr, optimizer_lr, loss_func, device)
            t_loss, _ = evaluate(model_lr, loader_train_lr, loss_func, device)
            v_loss, _ = evaluate(model_lr, loader_test_lr, loss_func, device)
            t_losses.append(t_loss)
            v_losses.append(v_loss)
            
        train_losses_lr[lr] = t_losses
        test_losses_lr[lr] = v_losses

    # Plot Training Loss for Q13
    plt.figure(figsize=(10, 5))
    for lr in learning_rates:
        plt.plot(range(1, epochs_q10 + 1), train_losses_lr[lr], label=f'Learning Rate {lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Q13: Training Loss vs. Epoch for Different Learning Rates')
    plt.legend()
    plt.show()

    # Plot Test Loss for Q13
    plt.figure(figsize=(10, 5))
    for lr in learning_rates:
        plt.plot(range(1, epochs_q10 + 1), test_losses_lr[lr], label=f'Learning Rate {lr}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Q13: Test Loss vs. Epoch for Different Learning Rates')
    plt.legend()
    plt.show()