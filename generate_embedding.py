import os
import numpy as np
import csv
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

# Initialize MTCNN (for face detection) and InceptionResnetV1 (for face embeddings)
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_embeddings(image_path):
    """
    Function to extract face embeddings from a given image.
    """
    try:
        img = Image.open(image_path)
        faces = mtcnn(img)  # Detect faces in the image
        
        if faces is not None:
            embeddings = resnet(faces)  # Generate embeddings
            return embeddings.detach().numpy()  # Return as NumPy array
        else:
            print(f"No face detected in {image_path}.")
            return None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images_in_folder(folder_path):
    """
    Process all images in a given folder and extract embeddings.
    """
    embeddings_dict = {}
    
    # Loop through all folders (each folder represents a person)
    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        
        if os.path.isdir(person_path):  # Check if it's a folder
            person_embeddings = []
            
            # Loop through all image files in the person's folder
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                
                # Make sure it's an image file
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                    print(f"Processing {image_file} from {person_folder}...")
                    embeddings = extract_embeddings(image_path)
                    
                    if embeddings is not None:
                        person_embeddings.append(embeddings.flatten())  # Flatten embeddings to 1D
            
            if person_embeddings:
                embeddings_dict[person_folder] = person_embeddings  # Store embeddings for this person
    
    return embeddings_dict

def save_embeddings_to_csv(embeddings_dict, filename):
    """
    Save embeddings to a CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        header = ['Person'] + [f"Embedding_{i+1}" for i in range(512)]  # 512 embedding dimensions
        writer.writerow(header)
        
        # Write embeddings for each person
        for person, embedding_list in embeddings_dict.items():
            for embedding in embedding_list:
                writer.writerow([person] + embedding.tolist())

if __name__ == "__main__":
    folder_path = r"C:\Users\ashifa ikram\Downloads\extracted_faces"  # Path to the folder containing extracted faces
    embeddings = process_images_in_folder(folder_path)
    
    # Save the embeddings to a CSV file
    save_embeddings_to_csv(embeddings, "face_embeddings.csv")

    print("Embeddings saved to face_embeddings.csv")
import os
import numpy as np
import csv
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch

mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def extract_embeddings(image_path):
    try:
        img = Image.open(image_path)
        faces = mtcnn(img)
        if faces is not None:
            embeddings = resnet(faces)
            return embeddings.detach().numpy()
        else:
            return None
    except Exception as e:
        return None

def process_images_in_folder(folder_path):
    embeddings_dict = {}
    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)
        if os.path.isdir(person_path):
            person_embeddings = []
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    embeddings = extract_embeddings(image_path)
                    if embeddings is not None:
                        person_embeddings.append(embeddings.flatten())
            if person_embeddings:
                embeddings_dict[person_folder] = person_embeddings
    return embeddings_dict

def save_embeddings_to_csv(embeddings_dict, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['Person'] + [f"Embedding_{i+1}" for i in range(512)]
        writer.writerow(header)
        for person, embedding_list in embeddings_dict.items():
            for embedding in embedding_list:
                writer.writerow([person] + embedding.tolist())

folder_path = r"C:\Users\ashifa ikram\Downloads\extracted_faces"
embeddings = process_images_in_folder(folder_path)
save_embeddings_to_csv(embeddings, "face_embeddings.csv")
