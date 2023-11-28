
import cv2
import os



labels = ["angry", "neutral", "bored", "focused"]


# Define the target image dimensions
target_width = 112
target_height = 112

#face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for name in labels:
    # Define the directory where your images are located
    input_dir = "Dataset/" + name
    output_dir = "cleaned-dataset/" + name

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
            # change color from RBG to Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                # Calculate the center of the face bounding box
                face_center_x = x + w // 2
                face_center_y = y + h // 2

                # Define the region to crop around the face center
                crop_width = int(w * 2)
                crop_height = int(h * 2)
                start_x = max(0, face_center_x - crop_width // 2)
                start_y = max(0, face_center_y - crop_height // 2)

                # Crop the image around the face center
                cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]

                # Resize the image to the target dimensions
                resized_image = cv2.resize(cropped_image, (target_width, target_height))

                # Save the standardized image to the output directory
                output_path = os.path.join(output_dir, file_name)
                cv2.imwrite(output_path, resized_image)

print("Data cleaning complete.")
