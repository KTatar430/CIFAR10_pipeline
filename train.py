# Training script for a convolutional neural network on the CIFAR-10 dataset
# The script trains the model for a specified number of epochs, evaluates it on the validation set
# after each epoch, and implements early stopping based on validation loss. Only the best model is saved.

import torch
import torch.nn as nn
import time

from data.dataset import get_dataloaders
from model.cnn import SimpleCNN

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # Use GPU if available
    best_val_loss = float("inf")
    counter = 0         # Counter for early stopping (number of epochs in a row with no improvement in validation loss)

    print("Starting training on device:", device)
    start_time = time.time()    # Track total training time

    # Hyperparameters
    epochs = 16
    batch_size = 32
    learning_rate = 0.001
    patience = 2        # Number of epochs in a row with validation loss smaller than best_val_loss required for early stopping

    trainloader, valloader, _, _ = get_dataloaders(batch_size)

    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # Training part
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()
        for images, labels in trainloader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = correct / total

        # Validation part
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in valloader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_fn(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_epoch_loss = val_loss / len(valloader)
        val_epoch_acc = val_correct / val_total
        epoch_duration = time.time() - epoch_start
        print(f"Epoch duration: {epoch_duration:.2f}s")

        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"Train Loss: {epoch_loss:.4f}, "
              f"Train Accuracy: {epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}, "
              f"Val Accuracy: {val_epoch_acc:.4f}")
        print("-" * 50)

        # Early stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved")
            print()
        else:
            counter += 1
            print("No improvement in validation loss")
            print()

        if counter >= patience:
            print("Early stopping triggered")
            print()
            break

    total_duration = time.time() - start_time
    print(f"Training completed in {total_duration:.2f} seconds.")
    print()
    print()


if __name__ == "__main__":
    train()
