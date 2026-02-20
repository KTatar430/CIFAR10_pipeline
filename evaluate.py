# Evaluation script for a convolutional neural network trained on the CIFAR-10 dataset
# The script loads the best model saved during training, evaluates it on the test set, and
# prints the test loss, test accuracy, and confusion matrix for the test set predictions.
# Additionally, it calculates and prints the accuracy for each class in the CIFAR-10 dataset.

import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import confusion_matrix

from data.dataset import get_dataloaders
from model.cnn import SimpleCNN


def evaluate():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    _, _, testloader, classes = get_dataloaders(batch_size)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("best_model.pth",
                                     map_location=device))

    loss_fn = nn.CrossEntropyLoss()

    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = correct / total
    test_loss = test_loss / len(testloader)

    print("-" * 40)
    print("TEST RESULTS")
    print("-" * 40)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({correct}/{total})")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    print("\nConfusion Matrix:")
    print(cm_df)

    # Every class accuracy
    print("\n" + "=" * 50)
    print("PER-CLASS ACCURACY")
    print("=" * 50)

    for i, class_name in enumerate(classes):
        class_correct = cm[i, i]
        class_total = cm[i].sum()
        class_acc = 100 * class_correct / class_total

        print(f"{class_name:15s}: "
              f"{class_acc:6.2f}% "
              f"({class_correct}/{class_total})")


if __name__ == "__main__":
    evaluate()
