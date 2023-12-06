#  necessary imports
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

import matplotlib.pyplot as plt
 
root_directoris = ['./Gender/Male', './Gender/Female','./Age/old', './Age/young']

path_to_model = './main_model.pth'

classes = ['angry', 'bored', 'focused', 'neutral']

def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=1)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=1)

    return accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1

# main model of CNN network
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


device = "cpu"
if (torch.cuda.is_available()):
    device = "cuda"

def load_dataset(test_data):

    transformation = transforms.Compose([

        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    testing_dataset = torchvision.datasets.ImageFolder(
        root=test_data,
        transform=transformation
    )

    test_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    return test_loader

loss_criteria = nn.CrossEntropyLoss()

# test function to test the datasets
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_criteria(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Testing set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_labels, all_predictions


def plot_confusion_matrix(true_label, predicted_label, model_name):
    true_label = np.array(true_label)
    predicted_label = np.array(predicted_label)
    conf_matrix = confusion_matrix(true_label, predicted_label)
    classes = ('angry', 'bored', 'focused', 'neutral')
    ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
loaded_model = CNN(num_classes=4).to(device)  # Assuming CNN is the model class
loaded_model.load_state_dict(torch.load(path_to_model))
loaded_model.eval()  # Set the model to evaluation mode

arr = ["Gender - Male", "Gender - Female", "Age - Old", "Age - Young"]
i = 0
for root_dir in root_directoris:
    name = f"{arr[i]}"
    print(name)
    loaded_model.to(device)
    test_loader = load_dataset(root_dir)
    labels, predictions = test(loaded_model, device, test_loader)
    i+=1
    plot_confusion_matrix(labels, predictions, name)

    # Calculate metrics for this fold
    fold_metrics = calculate_metrics(labels, predictions)
    main_model_accuracy, main_model_precision, main_model_recall, main_model_f1, main_model_micro_precision, main_model_micro_recall, main_model_micro_f1 = fold_metrics
    print("Model\t\tMacro Precision\tMacro Recall\tMacro F1\tAccuracy")
    print(f"{name}\t{main_model_precision:.4f}\t\t{main_model_recall:.4f}\t\t{main_model_f1:.4f}\t\t{main_model_accuracy:.4f}")