# COMP-6721-A.I.ducation-Analytics
COMP 6721 AAI Project to create A.I.ducation Analytics using CNN

# Project Description

- This A.I.ducation is a project to detect the student engagement during a lecture. 
- The main objective of this project is to use Convolutional Neural Networks (CNN) for facial expression detection and classify them into focused, bored, distracted, confused and neutral classes.
- Dataset to train this model is included in this Repository itself.

# How to Run?

1. Make sure you have Python and all the required libraries mentioned in Requirements.txt file installed on your system.

2. Clone the Repository onto your local machine.

3. First run dataCleaning.py, which will resize the images and convert them into grayscale.

4. Then run dataPartition.py, which will partition the dataset into training, testing and validation directories.

5. Lastly run dataVisualization.py, which will create the distribution bar chart for all classes in dataset, 5X5 grid image which shows some example images of each class and also its pixel intensities. These all images will be saved inside Docs folder.

6. In model_training.py, set your dataset path for training, testing and validation data and run that python file, which will train the CNN model on your training data and calculate accuracy for training, tetsing and validation sets for main model as well as its different varients.
After training it will save the model as with .pth extension.

7. In model_testing.py, we'll load the saved model by providing its path and try it on different images by setting the path of any image or you can give entire directory to predict the class of each image.

# Contributors

- Yash Chhelaiya
- Mihir Gediya
- Vanshika Singla