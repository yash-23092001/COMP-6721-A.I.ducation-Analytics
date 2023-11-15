import os
import shutil
import random

labels = ["bored", "angry", "focused", "neutral"]

def split(nameIn, source_dir):
    train_dir = "Dataset/train/" + nameIn
    test_dir = "Dataset/test/" + nameIn
    validate_dir = "Dataset/validation/" + nameIn

    # Create destination directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validate_dir, exist_ok=True)

    # List all files in the source directory
    file_list = os.listdir(source_dir)

    # Shuffle the file list randomly
    random.shuffle(file_list)

    # Calculate the split point based on 70% for training and 15% for testing and 15% for validation
    split_point_1 = int(0.7 * len(file_list))

     # Copy files to train directory
    for filename in file_list[:split_point_1]:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(train_dir, filename)
        shutil.copy(source_path, destination_path)

    #split remaining data into testeing and validation
    rem_data = file_list[split_point_1:]
    random.shuffle(rem_data)
    split_point_2 = int(0.5 * len(rem_data))

     # Copy files to test directory
    for filename in rem_data[:split_point_2]:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(test_dir, filename)
        shutil.copy(source_path, destination_path)

     # Copy files to validation directory
    for filename in rem_data[split_point_2:]:
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(validate_dir, filename)
        shutil.copy(source_path, destination_path)


for name in labels:
    # Define the source and destination directories
    dirS = "cleaned-dataset/" + name
    split(name, dirS)