
import os
import csv
import numpy as np
import pymysql
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import cv2
from torchvision import transforms

# -------------------- Model Initialization -------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -------------------- Preprocessing and Augmentation Function -------------------- #
def preprocess_and_extract_embeddings(image, augment_times=5):
    # Detect faces using MTCNN
    faces = mtcnn(image)
    if faces is None:
        print("😕 No face detected")
        return None

    embeddings_list = []
    
    # Augmentation pipeline
    augmentations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop((160, 160), scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])
    
    # Process each face detected
    for face_tensor in faces:
        face_img = face_tensor.permute(1, 2, 0).mul(255).byte().cpu().numpy()
        face_img = cv2.resize(face_img, (160, 160))
        
        # Augment the face multiple times
        for _ in range(augment_times):
            augmented = augmentations(face_img)
            augmented = augmented.unsqueeze(0).to(device).float()
            embedding = resnet(augmented).detach().cpu().numpy().flatten()

            if embedding.shape[0] >= 512:
                embeddings_list.append(embedding[:512])
    
    return embeddings_list if embeddings_list else None

# -------------------- Database Connection -------------------- #
def get_db_connection():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='',
        database='user_database'
    )

# -------------------- Extract Embeddings from Image -------------------- #
def extract_embeddings(image_path, augment_times=5):
    try:
        img = Image.open(image_path).convert('RGB')
        embeddings = preprocess_and_extract_embeddings(img, augment_times)
        
        if embeddings is None:
            print(f"😕 No face detected in: {image_path}")
            return None
        
        return embeddings

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None

# -------------------- Process Files from DB -------------------- #
def process_website_files(user_id=1):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT folders.folder_name, files.file_name, files.file_path
        FROM files 
        JOIN folders ON files.folder_id = folders.id
        WHERE folders.user_id = %s
    """, (user_id,))
    rows = cursor.fetchall()

    base_folder = os.path.join(os.getcwd(), 'uploads')  # uploads/user_id/folder_name/file_name
    embeddings_list = []

    for folder_name, file_name, file_path in rows:
        print(f"\n📂 Folder: {folder_name}")
        print(f"🖼️ File: {file_name}")

        folder_path = os.path.join(base_folder, str(user_id), folder_name)
        expected_path = os.path.join(folder_path, file_name)

        print(f"📍 Expected Path: {expected_path}")

        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        matched_file = next(
            (f for f in os.listdir(folder_path) if f.lower() == file_name.lower()), None
        )

        if matched_file:
            full_path = os.path.join(folder_path, matched_file)
            print(f"✅ Processing file: {full_path}")

            embeddings = extract_embeddings(full_path)

            if embeddings:
                for emb in embeddings:
                    embeddings_list.append([
                        folder_name,  # Class
                        os.path.splitext(file_name)[0]  # Person name (without extension)
                    ] + emb.tolist())
            else:
                print(f"⚠️ No valid embeddings for: {matched_file}")
        else:
            print(f"❌ File not found (case mismatch maybe): {file_name}")

    cursor.close()
    conn.close()
    return embeddings_list

# -------------------- Save to CSV -------------------- #
def save_to_csv(embeddings_list, filename='cleaned_face_embeddings.csv'):
    if not embeddings_list:
        print("🚫 No embeddings to save.")
        return

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Person'] + [f'Embedding_{i+1}' for i in range(512)])
        for row in embeddings_list:
            writer.writerow(row)

    print(f"\n✅ Embeddings saved to: {filename}")

# -------------------- Combined Callable Function -------------------- #
def run_embedding_extraction(user_id=1, output_file='cleaned_face_embeddings.csv'):
    print("🔍 Starting embedding extraction for website files...")
    embeddings = process_website_files(user_id=user_id)
    if embeddings:
        save_to_csv(embeddings, filename=output_file)
    else:
        print("❌ No embeddings extracted. Check image quality and face visibility.")

