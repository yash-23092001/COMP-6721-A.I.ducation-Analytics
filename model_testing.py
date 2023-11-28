import numpy as np
import os

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision.io import read_image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# path to datasets
train_data = '/content/Dataset/train'
test_data = '/content/Dataset/test'
path_to_model = '/content/drive/MyDrive/main_model.pth'
classes = sorted(os.listdir(train_data))

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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((112,112))
])

# Load the trained model
loaded_model = CNN(num_classes=4).to(device)  # Assuming CNN is the model class
loaded_model.load_state_dict(torch.load(path_to_model))
loaded_model.eval()  # Set the model to evaluation mode

# Load and preprocess the input image


from PIL import Image
file = input("File or Dataset? (F/D)? ")


if file == 'D' or file == 'd':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded_model.to(device)
    test_loader = load_dataset(test_data)
    labels, predictions = test(loaded_model, device, test_loader)
    plot_confusion_matrix(labels, predictions, "Main Model")

elif file == 'F' or file == 'f':
    # image_path = '/content/Dataset/test/focused/FO_1000.jpg'
    image_path = '/content/Dataset/test/bored/B_2873.jpg'
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