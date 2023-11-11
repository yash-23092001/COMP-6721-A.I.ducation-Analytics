import os
import shutil
import random

labels = ["bored", "angry", "focused", "neutral"]

def split(nameIn, source_dir):
    train_dir = "partitioned-dataset/train/" + nameIn
    test_dir = "partitioned-dataset/test/" + nameIn

    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # List all files in the source directory
    file_list = os.listdir(source_dir)

    # Shuffle the file list randomly
    random.shuffle(file_list)

    # Calculate the split point based on 80% for training and 20% for testing
    split_point = int(0.8 * len(file_list))

    # Copy files to train directory
    for filename in file_list[:split_point]:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(train_dir, filename)
        shutil.copy(source_path, destination_path)

    # Copy files to test directory
    for filename in file_list[split_point:]:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(test_dir, filename)
        shutil.copy(source_path, destination_path)


for name in labels:
    # Define the source and destination directories
    dirS = "cleaned-dataset/" + name
    split(name, dirS)