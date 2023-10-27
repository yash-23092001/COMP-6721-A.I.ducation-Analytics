
import cv2
import numpy as np
import os
import random


dataset = ["train", "test"]
labels = ["bored", "confused", "distracted", "focused", "neutral"]


# Define the target image dimensions
target_width = 256
target_height = 256

for dir in dataset:
    for name in labels:
        # Define the directory where your images are located
        input_dir = "partitioned-dataset/" + dir + "/" + name
        output_dir = "standardized-dataset/" + dir + "/" + name

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List the files in the input directory
        input_files = os.listdir(input_dir)

        # Loop through each image and perform standardization
        for file_name in input_files:
            # Load the image using OpenCV
            image_path = os.path.join(input_dir, file_name)
            image = cv2.imread(image_path)

            if not image is None:
                # Resize the image to the target dimensions
                image = cv2.resize(image, (target_width, target_height))

                # Apply brightness adjustment
                image = cv2.convertScaleAbs(image, alpha=1, beta=0)

                # change color from RBG to Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply minor cropping (10 pixels from each side)
                image = image[10:target_height - 10, 10:target_width - 10]

                # Save the standardized image to the output directory
                output_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_path, image)

print("Standardization complete.")
