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

root_dir = './cleaned-dataset'

path_to_model = './main_model.pth'

classes = os.listdir(root_dir)

file_paths = {c: [] for c in classes}

for label in classes:
    class_dir = os.path.join(root_dir, label)
    file_names = os.listdir(class_dir)
    file_paths[label] = [os.path.join(class_dir, file) for file in file_names]

train_files, val_files, test_files = {}, {}, {}

for label, paths in file_paths.items():
    train_val_files, test_files[label] = train_test_split(
        paths, test_size=0.15, random_state=42)
    train_files[label], val_files[label] = train_test_split(
        train_val_files, test_size=0.1765, random_state=42)  # 70% of 85% = ~60%

test_set = [file for files in test_files.values() for file in files]

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


def load_dataset(test_data):
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    test_dataset = CustomFacialImageDataset(test_set, transform=transformation)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    return test_loader


test_loader = load_dataset(test_set)
batch_size = test_loader.batch_size


loss_criteria = nn.CrossEntropyLoss()

# test function to test the datasets

target_to_index = {"angry": 0, "bored": 1, "focused": 2, "neutral": 3}


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
            targets = torch.tensor([target_to_index[target]
                                   for target in targets]).to(device)

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

# Load and preprocess the input image


file = input("File or Dataset or 10 random Images? (F/D/R)? ")
# file = 'D'

if file == 'D' or file == 'd':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model.to(device)
    test_loader = load_dataset(test_set)
    labels, predictions = test(loaded_model, device, test_loader)
    plot_confusion_matrix(labels, predictions, "Main Model")

elif file == 'F' or file == 'f':
    image_path = '/content/drive/MyDrive/img.jpeg.jpg'
    # image_path = '/content/Dataset/test/focused/FO_1000.jpg'
    # image_path = '/content/Dataset/test/bored/B_2873.jpg'
    # image_path = '/content/cleaned-dataset/focused/FO_2055.jpg'
    # image_path = '/content/cleaned-dataset/neutral/N_1.jpg'
    input_image = Image.open(image_path).convert("RGB")
    input_image = transform(input_image).unsqueeze(0).to(device)

    # Make the prediction
    with torch.no_grad():
        output = loaded_model(input_image)

    # Get the predicted class
    _, predicted_class = output.max(1)

    # Print or use the predicted class
    print("Predicted Class:", classes[predicted_class.item()])

elif file == 'r' or file == 'R':
    # Get some random samples from the test_loader
    num_samples = 10
    sample_indices = random.sample(
        range(len(test_loader.dataset)), num_samples)
    sample_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_loader.dataset, sample_indices),
        batch_size=num_samples,
        shuffle=False
    )

    # Get predictions for the random samples
    with torch.no_grad():
        for data, targets in sample_loader:
            data = data.to(device)
            targets = torch.tensor([target_to_index[target]
                                   for target in targets]).to(device)
            output = loaded_model(data)
            _, predicted = torch.max(output, 1)

    # Visualize the random samples along with predictions
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

    for i in range(num_samples):
        image, true_label, predicted_label = data[i].cpu().numpy().transpose(
            (1, 2, 0)), classes[targets[i]], classes[predicted[i]]
        axes[i].imshow(image)
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}')
        axes[i].axis('off')

    plt.show()
