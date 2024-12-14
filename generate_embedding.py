import os
import numpy as np
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
    img = Image.open(image_path)
    # Detect faces in the image
    faces = mtcnn(img)  # MTCNN now returns only the faces
    
    if faces is not None:
        embeddings = resnet(faces)  # Generate embeddings
        return embeddings.detach().numpy()  # Return as NumPy array
    else:
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
                        person_embeddings.append(embeddings)
            
            if person_embeddings:
                embeddings_dict[person_folder] = np.array(person_embeddings)  # Store embeddings for this person
    
    return embeddings_dict

if __name__ == "__main__":
    folder_path = r"C:\Users\ashifa ikram\Downloads\extracted_faces"  # Correct path to your folder
    embeddings = process_images_in_folder(folder_path)
    
    # Flatten embeddings and print them
    for person, embedding_array in embeddings.items():
        print(f"Embeddings for {person}: {embedding_array.shape}")
        
        # Flatten embeddings (remove the extra dimension)
        flat_embeddings = embedding_array.squeeze()  # This removes the unnecessary singleton dimension (1)

        # Now flat_embeddings will be of shape (n, 512), where n is the number of images
        print(f"Flattened embeddings for {person}: {flat_embeddings.shape}")
        
        # You can use flat_embeddings for further processing like classificapiption or comparison
