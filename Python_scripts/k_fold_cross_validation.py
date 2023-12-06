import numpy as np
import pandas as pd
import os
import random

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold


import matplotlib.pyplot as plt

root_dir = './cleaned-dataset'

classes = os.listdir(root_dir)

file_paths = {c: [] for c in classes}

for label in classes:
    class_dir = os.path.join(root_dir, label)
    file_names = os.listdir(class_dir)
    file_paths[label] = [os.path.join(class_dir, file) for file in file_names]

train_files, val_files, test_files = {}, {}, {}

for label, paths in file_paths.items():
    train_val_files, test_files[label] = train_test_split(paths, test_size=0.15, random_state=42)
    train_files[label], val_files[label] = train_test_split(train_val_files, test_size=0.1765, random_state=42)  # 70% of 85% = ~60%

train_set = [file for files in train_files.values() for file in files]
val_set = [file for files in val_files.values() for file in files]
test_set = [file for files in test_files.values() for file in files]


device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda"

class CustomFacialImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = img_path.split('/')[-2]
        return image, label

def load_dataset(train_data, validation_data, test_data):
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = CustomFacialImageDataset(train_set, transform=transformation)
    validation_dataset = CustomFacialImageDataset(val_set, transform=transformation)
    test_dataset = CustomFacialImageDataset(test_set, transform=transformation)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    return train_loader, validation_loader, test_loader


train_loader, validation_loader, test_loader = load_dataset(
    train_set, val_set, test_set)
batch_size = train_loader.batch_size

# main model of the CNN


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(200704, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

model = CNN(num_classes=4)
target_to_index= {"angry":0,"bored":1,"focused":2,"neutral":3}
loss_criteria = nn.CrossEntropyLoss()
epochs = 15
patience = 3

def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    correct = 0
    total = 0



    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, ((data), targets) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data = data.to(device)
        targets = torch.tensor([target_to_index[target] for target in targets]).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, targets)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / (batch_idx+1)
    train_loss += loss.item()

    accuracy = 100. * correct / total
    print('Training set: Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%)'.format(
        avg_loss, correct, total, accuracy))

    return avg_loss


def validate(model, device, validation_loader):
    model.eval()
    validation_loss = 0
    correct = 0



    with torch.no_grad():
        batch_count = 0
        for data, targets in validation_loader:
            batch_count += 1
            data = data.to(device)
            targets = torch.tensor([target_to_index[target] for target in targets]).to(device)

            output = model(data)

            validation_loss += loss_criteria(output, targets).item()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(targets == predicted).item()

    avg_loss = validation_loss / batch_count


    print('Validation set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)'.format(
        avg_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))
    return avg_loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        batch_count = 0
        for data, targets in test_loader:
            batch_count += 1
            data = data.to(device)
            targets = torch.tensor([target_to_index[target] for target in targets]).to(device)

            output = model(data)

            test_loss += loss_criteria(output, targets).item()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(targets == predicted).item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    avg_loss = test_loss / batch_count
    print('Testing set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_labels, all_predictions


# Define the number of folds
num_folds = 10

# Create a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize lists to store metrics for each fold
all_fold_metrics = []
patience = 3

def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=1)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=1)

    return accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1

# Iterate over the folds
for fold, (train_index, val_index) in enumerate(kf.split(train_set)):
    print(f"Fold {fold + 1}/{num_folds}")

    # Split the data into training and validation sets for this fold
    fold_train_set = [train_set[i] for i in train_index]
    fold_val_set = [train_set[i] for i in val_index]

    # Create data loaders for this fold
    fold_train_loader, fold_val_loader, _ = load_dataset(
        fold_train_set, fold_val_set, test_set)

    # Initialize the model for this fold
    fold_model = CNN(num_classes=4)

    # Initialize optimizer and loss function
    fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.00001)
    fold_loss_criteria = nn.CrossEntropyLoss()

    best_valid_loss = float('inf')
    current_patience = 0

    # Train the model for this fold
    for epoch in range(1, epochs + 1):
        train(fold_model, device, fold_train_loader, fold_optimizer, epoch)
        loss = validate(fold_model, device, fold_val_loader)
        if loss < best_valid_loss:
            best_valid_loss = loss
            current_patience = 0
        else:
            current_patience += 1

        if current_patience >= patience:
            print(f'Early stopping at epoch {epoch}.')
            break

    # Test the model on the test set for this fold
    fold_labels, fold_predictions = test(fold_model, device, test_loader)

    # Calculate metrics for this fold
    fold_metrics = calculate_metrics(fold_labels, fold_predictions)
    all_fold_metrics.append(fold_metrics)
    main_model_accuracy, main_model_precision, main_model_recall, main_model_f1, main_model_micro_precision, main_model_micro_recall, main_model_micro_f1 = fold_metrics
    print("Model\t\tMacro Precision\tMacro Recall\tMacro F1\tMicro Precision\tMicro Recall\tMicro F1\tAccuracy")
    print(f"Main Model\t{main_model_precision:.4f}\t\t{main_model_recall:.4f}\t\t{main_model_f1:.4f}\t\t{main_model_micro_precision:.4f}\t\t{main_model_micro_recall:.4f}\t\t{main_model_micro_f1:.4f}\t\t{main_model_accuracy:.4f}")
    model_path = f"{fold+1}_model.pth"

    
# Average metrics over all folds
average_metrics = np.mean(all_fold_metrics, axis=0)

# Print or store the average metrics as needed
print("Average Metrics over 10 Folds:")
print("Model\t\tMacro Precision\tMacro Recall\tMacro F1\tMicro Precision\tMicro Recall\tMicro F1\tAccuracy")
print(f"Main Model\t{average_metrics[1]:.4f}\t\t{average_metrics[2]:.4f}\t\t{average_metrics[3]:.4f}\t\t{average_metrics[4]:.4f}\t\t{average_metrics[5]:.4f}\t\t{average_metrics[6]:.4f}\t\t{average_metrics[0]:.4f}")