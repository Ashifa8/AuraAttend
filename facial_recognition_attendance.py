import os
import cv2  # OpenCV for image processing
import numpy as np

# Define the path to the folder containing images
image_folder = r'C:\Users\qc\Desktop\New folder (3)\AuraAttend\image'  # Use raw string

# List to store all image data as numpy arrays
image_data_list = []

# Function to load all images from a folder
def load_images_from_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        # Read the image using OpenCV (in grayscale mode)
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Print the shape of the loaded image
            print(f"Loaded {filename} with shape: {image.shape}")
            # Resize image to a common size, e.g., 256x256
            image = cv2.resize(image, (256, 256))
            # Append image data to the list
            image_data_list.append(image)

# Function to calculate mean and median of images
def calculate_mean_median(images):
    # Stack all images into one 3D array (assuming all images have the same dimensions)
    image_stack = np.array(images)
    
    # Calculate the mean and median across the image stack
    mean_image = np.mean(image_stack, axis=0)
    median_image = np.median(image_stack, axis=0)
    
    return mean_image, median_image

# Main execution
load_images_from_folder(image_folder)

if len(image_data_list) > 0:
    mean_image, median_image = calculate_mean_median(image_data_list)
    
    print("Mean pixel values calculated for all images.")
    print("Median pixel values calculated for all images.")
    
    # Save mean and median images to files (optional)
    cv2.imwrite('mean_image.jpg', mean_image)
    cv2.imwrite('median_image.jpg', median_image)
    
    # Display the mean and median images
    cv2.imshow('Mean Image', mean_image.astype(np.uint8))
    cv2.imshow('Median Image', median_image.astype(np.uint8))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No images found in the folder.")
