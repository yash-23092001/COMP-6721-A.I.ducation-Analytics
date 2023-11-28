#  necessary imports
import numpy as np
import pandas as pd
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.io import read_image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

os.makedirs("Confusion_Matrix")
os.makedirs("models")

# dataset paths
train_data = '/content/Dataset/train'
validation_data = '/content/Dataset/validation'
test_data = '/content/Dataset/test'

# classes of the model
classes = sorted(os.listdir(train_data))
print(classes)

# function to load the dataset
# that will randomly transform the images, to train model perfectly
def load_dataset(train_data, validation_data, test_data):
    transformation = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    dataset = torchvision.datasets.ImageFolder(
        root=train_data,
        transform=transformation
    )

    validating_dataset = torchvision.datasets.ImageFolder(
        root=validation_data,
        transform=transformation
    )
    testing_dataset = torchvision.datasets.ImageFolder(
        root=test_data,
        transform=transformation
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    validation_loader = torch.utils.data.DataLoader(
        validating_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        testing_dataset,
        batch_size=32,
        num_workers=0,
        shuffle=True
    )

    return train_loader, validation_loader, test_loader


train_loader, validation_loader, test_loader = load_dataset(
    train_data, validation_data, test_data)
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

# variant1 of main model of the CNN

class CNNVariant1(nn.Module):
    def __init__(self, num_classes):
        super(CNNVariant1, self).__init__()
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
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(401408, 1024),
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

# variant2 of main model of the CNN

class CNNVariant2(nn.Module):
    def __init__(self, num_classes):
        super(CNNVariant2, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32,
                      kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(53824, 1024),
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

model = CNN(num_classes=4).to(device)
modelv1 = CNNVariant1(num_classes=4).to(device)
modelv2 = CNNVariant2(num_classes=4).to(device)

models = [["Main", model], ["Variant 1", modelv1], [
    "Variant 2", modelv2]]
print(device)


def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    correct = 0
    total = 0



    print("Epoch:", epoch)
    # Process the images in batches
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_criteria(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

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
        for data, target in validation_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            output = model(data)

            validation_loss += loss_criteria(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

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
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_criteria(output, target).item()

            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    avg_loss = test_loss / batch_count
    print('Testing set: Average loss: {:.6f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_labels, all_predictions


optimizer = optim.Adam(model.parameters(), lr=0.00001)

loss_criteria = nn.CrossEntropyLoss()

epochs = 15
patience = 3


print('Training on', device)
flag = 1
for m in models:
    best_valid_loss = float('inf')
    current_patience = 0
    print("For Model: ", m[0])
    for epoch in range(1, epochs + 1):
        train(m[1], device, train_loader, optimizer, epoch)

        flag -= 1
        loss = validate(m[1], device, validation_loader)

        test(m[1], device, test_loader)



        if loss < best_valid_loss:
            best_valid_loss = loss
            current_patience = 0
        # Save the best model
            if flag == 0:
                torch.save(model.state_dict(), 'best_model.pth')
        else:
            current_patience += 1

        if current_patience >= patience:
            print(f'Early stopping at epoch {epoch}.')
            break

    model_path = f"{m[0].lower()}_model.pth"
    torch.save(m[1].state_dict(), model_path)

# function to calculate metrics


def calculate_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=1)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='micro', zero_division=1)

    return accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1


# Evaluate the Main Model
main_model_labels, main_model_predictions = test(model, device, test_loader)
main_model_metrics = calculate_metrics(
    main_model_labels, main_model_predictions)

# Evaluate Variant 1
variant1_labels, variant1_predictions = test(modelv1, device, test_loader)
variant1_metrics = calculate_metrics(variant1_labels, variant1_predictions)

# Evaluate Variant 2
variant2_labels, variant2_predictions = test(modelv2, device, test_loader)
variant2_metrics = calculate_metrics(variant2_labels, variant2_predictions)


main_model_accuracy, main_model_precision, main_model_recall, main_model_f1, main_model_micro_precision, main_model_micro_recall, main_model_micro_f1 = main_model_metrics
variant1_accuracy, variant1_precision, variant1_recall, variant1_f1, variant1_micro_precision, variant1_micro_recall, variant1_micro_f1 = variant1_metrics
variant2_accuracy, variant2_precision, variant2_recall, variant2_f1, variant2_micro_precision, variant2_micro_recall, variant2_micro_f1 = variant2_metrics

# Print or store the metrics as needed
print("Model\t\tMacro Precision\tMacro Recall\tMacro F1\tMicro Precision\tMicro Recall\tMicro F1\tAccuracy")
print(f"Main Model\t{main_model_precision:.4f}\t\t{main_model_recall:.4f}\t\t{main_model_f1:.4f}\t\t{main_model_micro_precision:.4f}\t\t{main_model_micro_recall:.4f}\t\t{main_model_micro_f1:.4f}\t\t{main_model_accuracy:.4f}")
print(f"Variant 1\t{variant1_precision:.4f}\t\t{variant1_recall:.4f}\t\t{variant1_f1:.4f}\t\t{variant1_micro_precision:.4f}\t\t{variant1_micro_recall:.4f}\t\t{variant1_micro_f1:.4f}\t\t{variant1_accuracy:.4f}")
print(f"Variant 2\t{variant2_precision:.4f}\t\t{variant2_recall:.4f}\t\t{variant2_f1:.4f}\t\t{variant2_micro_precision:.4f}\t\t{variant2_micro_recall:.4f}\t\t{variant2_micro_f1:.4f}\t\t{variant2_accuracy:.4f}")

# function to plot confusion matrix
def plot_confusion_matrix(true_label, predicted_label, model_name):

    true_label = np.array(true_label)
    predicted_label = np.array(predicted_label)
    conf_matrix = confusion_matrix(true_label, predicted_label)
    classes = ('angry', 'bored', 'focused', 'neutral')
    ConfusionMatrixDisplay(conf_matrix, display_labels=classes).plot()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'{model_name}.png')


plot_confusion_matrix(main_model_labels, main_model_predictions, models[0][0])
plot_confusion_matrix(variant1_labels, variant1_predictions, models[1][0])
plot_confusion_matrix(variant2_labels, variant2_predictions, models[2][0])
