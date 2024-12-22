import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

image_data_list = []
mtcnn = MTCNN(keep_all=True)
feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gamma_correction(image, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def histogram_equalization(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(image_gray)

def load_and_process_images(main_folder_path, output_base_folder, required_size=(256, 256), face_output_size=(160, 160), prob_threshold=0.9):
    for root, dirs, files in os.walk(main_folder_path):
        if root == main_folder_path:
            continue
        
        person_name = os.path.basename(root)
        person_output_folder = os.path.join(output_base_folder, person_name)
        create_dir(person_output_folder)

        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                image = cv2.imread(file_path)
                if image is not None:
                    image_corrected = gamma_correction(image)
                    image_equalized = histogram_equalization(image_corrected)
                    img_rgb = cv2.cvtColor(image_equalized, cv2.COLOR_BGR2RGB)
                    faces, probs = mtcnn.detect(img_rgb)

                    if faces is not None:
                        for i, (x1, y1, x2, y2) in enumerate(faces):
                            if probs[i] < prob_threshold:
                                continue

                            face = image[int(y1):int(y2), int(x1):int(x2)]
                            if face.size == 0:
                                continue

                            face_resized = cv2.resize(face, face_output_size)
                            output_file_path = os.path.join(
                                person_output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg"
                            )
                            cv2.imwrite(output_file_path, face_resized)
                            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                            image_data_list.append(face_gray)

def calculate_mean_median(images):
    image_stack = np.array(images)
    mean_image = np.mean(image_stack, axis=0)
    median_image = np.median(image_stack, axis=0)
    return mean_image, median_image

main_folder_path = r'C:\Users\ashifa ikram\Downloads\pictures'
output_base_folder = r'C:\Users\ashifa ikram\Downloads\extracted_faces'
create_dir(output_base_folder)
load_and_process_images(main_folder_path, output_base_folder)

if len(image_data_list) > 0:
    mean_image, median_image = calculate_mean_median(image_data_list)
    print("Mean pixel values calculated for all images.")
    print("Median pixel values calculated for all images.")
else:
    print("No faces found in the folder.")
