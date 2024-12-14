import os
import cv2  # OpenCV for image processing
import numpy as np

# List to store all image data as numpy arrays
image_data_list = []

# Path to the Haar Cascade for face detection
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Function to create a directory if it doesn't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to load and process images, saving extracted faces in person-specific folders
def load_and_process_images(main_folder_path, output_base_folder, required_size=(256, 256), face_output_size=(160, 160)):
    # Iterate through each subfolder in the main folder
    for root, dirs, files in os.walk(main_folder_path):
        # Skip the root folder itself
        if root == main_folder_path:
            continue
        
        # Get the name of the person (subfolder name)
        person_name = os.path.basename(root)
        # Create a corresponding output folder for the person
        person_output_folder = os.path.join(output_base_folder, person_name)
        create_dir(person_output_folder)

        for filename in files:
            file_path = os.path.join(root, filename)
            # Check if the file is an image based on its extension
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                # Read the image using OpenCV
                image = cv2.imread(file_path)
                if image is not None:
                    # Convert the image to grayscale for face detection
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Resize image for mean/median calculation
                    resized_image = cv2.resize(gray_image, required_size)
                    # Append resized image data to the list
                    image_data_list.append(resized_image)

                    # Perform face detection and extraction
                    faces = face_cascade.detectMultiScale(
                        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )
                    for i, (x, y, w, h) in enumerate(faces):
                        # Extract and resize the detected face
                        face = image[y:y+h, x:x+w]
                        face_resized = cv2.resize(face, face_output_size)
                        # Save the extracted face in the person's folder
                        output_file_path = os.path.join(
                            person_output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                        )
                        cv2.imwrite(output_file_path, face_resized)

# Function to calculate mean and median of images
def calculate_mean_median(images):
    # Stack all images into one 3D array (assuming all images have the same dimensions)
    image_stack = np.array(images)
    # Calculate the mean and median across the image stack
    mean_image = np.mean(image_stack, axis=0)
    median_image = np.median(image_stack, axis=0)
    return mean_image, median_image

# Main execution
main_folder_path = r'C:\Users\ashifa ikram\Downloads\pictures'  # Replace with the correct path to your main folder
output_base_folder = r'C:\Users\ashifa ikram\Downloads\extracted_faces'  # Base folder for all extracted faces
create_dir(output_base_folder)

# Load and process images
load_and_process_images(main_folder_path, output_base_folder)

if len(image_data_list) > 0:
    # Calculate mean and median for all resized images
    mean_image, median_image = calculate_mean_median(image_data_list)
    print("Mean pixel values calculated for all images.")
    print("Median pixel values calculated for all images.")
else:
    print("No images found in the folder.")
