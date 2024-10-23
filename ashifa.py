import os
import numpy as np
import cv2

# Function to read images, convert to grayscale, and resize them to a standard size
def load_images_from_folder(folder_path, required_size=(160, 160)):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is an image (e.g., has a .jpg or .png extension)
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Read the image using OpenCV
            image = cv2.imread(file_path)
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize the grayscale image to the required size (160x160)
            gray_image = cv2.resize(gray_image, required_size)
            # Normalize pixel values to the range [0, 1]
            gray_image = gray_image.astype('float32') / 255.0
            images.append(gray_image)
    return np.array(images)
# Specify the path to the folder containing your 9 images
folder_path = r'C:\Users\ashifa ikram\Downloads\ashifa'  # Use a raw string to handle backslashes

# Load and preprocess the images
images = load_images_from_folder(folder_path)

# Calculate the mean of the images across all pixel values
mean_image = np.mean(images, axis=0)

# Calculate the median of the images across all pixel values
median_image = np.median(images, axis=0)

print(f"Loaded {images.shape[0]} images from the folder.")
print("Mean image shape:", mean_image.shape)
print("Median image shape:", median_image.shape)
