import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

labels = ["bored", "angry", "focused", "neutral"]
train_img_counts = []
test_img_counts = []
validation_img_counts = []

for name in labels:
    train_dir_path = "Dataset/train/" + name
    test_dir_path = "Dataset/test/" + name
    validation_dir_path = "Dataset/validation/" + name

    train_file_count = len([f for f in os.listdir(train_dir_path)])
    test_file_count = len([f for f in os.listdir(test_dir_path)])
    validation_file_count = len([f for f in os.listdir(validation_dir_path)])

    train_img_counts.append(train_file_count)
    test_img_counts.append(test_file_count)
    validation_img_counts.append(validation_file_count)

os.mkdir("Docs")

#  Create a stacked bar graph
plt.bar(labels, train_img_counts, label=f'Training Images {train_img_counts}')
plt.bar(labels, test_img_counts, label=f'Testing Images {test_img_counts}', bottom=train_img_counts)
plt.bar(labels, validation_img_counts, label=f'Validation Images {validation_img_counts}', bottom=np.array(train_img_counts) + np.array(test_img_counts))
plt.xlabel('Facial Expressions')
plt.ylabel('Number of Images')
plt.title('Bar Graph of number of images for each facial expressions')

# Add a legend
plt.legend()

# save the bar graph
plt.savefig("Docs/Distribution_Bar_Chart.jpg")


# Create a 5x5 grid of subplots
fig, axes = plt.subplots(5, 5, figsize=(10, 10))

# list of randomply selected images
selected_rendom_images = []

# Load and display different images in each subplot
for i in range(len(labels)):
    img_dir = "Dataset/train/" + labels[i]
    file_list = os.listdir(img_dir)
    for j in range(5):
        # Load the images you want to display
        img = os.path.join(img_dir, file_list[random.randint(0, 501)])
        if not img is None:
            selected_rendom_images.append(img)
            image = cv2.imread(img)

            # Display the image in the current subplot
            axes[i, j].imshow(image) 
            axes[i, j].set_title(labels[i])

# Hide axis labels and ticks
for ax in axes.ravel():
    ax.axis('off')

# Adjust layout
plt.tight_layout()

# save the grid of images
plt.savefig("Docs/grid_images.jpg")


# pixel intensity distribution
pixel_intensities = []

for image in selected_rendom_images:
    sel_img = cv2.imread(image)
    pixel_intensities.append(sel_img.flatten())

plt.figure(figsize=(10, 10))

for i, intensities in enumerate(pixel_intensities):
    plt.subplot(5, 5, i+1) 
    plt.hist(intensities, bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    plt.title(f'Image {i + 1}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("Docs/pixel_intensity.jpg")
