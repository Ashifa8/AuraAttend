from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the stored embeddings and names from the CSV file
STORED_EMBEDDINGS_PATH = "face_embeddings.csv"
df = pd.read_csv(STORED_EMBEDDINGS_PATH)
stored_embeddings = df.drop('Person', axis=1).values  # Known face embeddings
names_list = df['Person'].values  # Corresponding names

# Load MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the feature extractor (InceptionResnetV1)
feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"status": "error", "message": "No image data received."})

        # Decode the base64-encoded image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Error decoding the image."})

        # Convert image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces, _ = mtcnn.detect(img_rgb)

        if faces is None or len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected."})

        # Prepare response data
        results = []

        for (x1, y1, x2, y2) in faces:
            # Extract and preprocess the face
            face = img[int(y1):int(y2), int(x1):int(x2)]
            face_resized = cv2.resize(face, (160, 160))
            face_normalized = face_resized / 255.0
            face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0).float()

            # Extract embedding for the detected face
            embedding = feature_extractor(face_tensor).detach().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding

            # Compare the extracted embedding with stored embeddings
            best_similarity = -1
            recognized_name = "Unknown"

            for stored_embedding, name in zip(stored_embeddings, names_list):
                similarity = cosine_similarity(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    recognized_name = name

            # Check if the best similarity meets the threshold
            THRESHOLD = 0.6  # Adjust as needed
            if best_similarity < THRESHOLD:
                recognized_name = "Unknown"

            results.append({"name": recognized_name, "similarity": best_similarity})

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
