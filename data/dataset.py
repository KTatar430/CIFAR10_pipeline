# Getting data loaders for training, validation, and testing on the CIFAR-10 dataset
# The training set is augmented with random cropping and horizontal flipping, 
# Validation and test sets are not augmented

import torch
import torchvision
import torchvision.transforms as transforms

def get_dataloaders(batch_size=32):

    # Separate transforms for training (with augmentation) and test/validation (no augmentation)
    # Augmenting training data with random cropping and horizontal flipping to improve generalization
    # Validation and test data are only normalized without augmentation to evaluate true performance of the model
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                   # Random crop with padding (32x32 images)
        transforms.RandomHorizontalFlip(),                                      # 50% chance of horizontal flip
        transforms.ToTensor(),                                                  # scaling to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                  # scaling to [-1, 1]
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),                                                  # scaling to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                  # scaling to [-1, 1]
    ])

    # Downloads data if not already present, and loads it as a PyTorch dataset
    full_trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=None)
    test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=None)


    # Split training set into train and validation (80-20) randomly 
    # (every class should be represented in both sets due to large amount of data)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size

    # use fixed seed for reproducible split
    generator = torch.Generator().manual_seed(42)
    trainset, valset = torch.utils.data.random_split(full_trainset, [train_size, val_size], generator=generator)


    # Augment train and validation sets with the appropriate transforms (train gets augmentation, val gets no augmentation)
    train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./dataset',train=True,download=False, transform=test_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=test_transform)

    trainset = torch.utils.data.Subset(train_dataset, trainset.indices)
    valset = torch.utils.data.Subset(val_dataset, valset.indices)

    # Create data loaders for training, validation, and testing with the specified batch size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classes = train_dataset.classes

    return trainloader, valloader, testloader, classes
