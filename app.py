from flask import Flask, render_template, request, redirect, url_for, jsonify, session, make_response
from flask_cors import CORS
import mysql.connector
from datetime import datetime
import bcrypt  # Password is hashed using bcrypt before being stored in the database for better security
import os
import base64
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import pandas as pd
import csv

# Initialize the Flask app
app = Flask(_name_)
CORS(app, supports_credentials=True)  # Allows cross-origin requests with credentials

# Secret key for session management
app.secret_key = "your_secret_key"  # Set a secure secret key for session management

# Database connection function
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Replace with your MySQL username
        password="",  # Replace with your MySQL password
        database="user_database"  # Replace with your database name
    )

# Load MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the feature extractor (InceptionResnetV1)
feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()

# Load stored embeddings and names from the CSV file
STORED_EMBEDDINGS_PATH = "face_embeddings.csv"
if os.path.exists(STORED_EMBEDDINGS_PATH):
    df = pd.read_csv(STORED_EMBEDDINGS_PATH)
    stored_embeddings = df.drop('Person', axis=1).values.tolist()  # Convert to list
    names_list = df['Person'].values.tolist()  # Convert to list
else:
    stored_embeddings = []
    names_list = []

# Cosine similarity function for face recognition
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    return similarity

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']   # Fix: Match frontend key
    password = data['password']   # Fix: Match frontend key

    conn = get_db_connection()
    cursor = conn.cursor()

    # Check if username already exists
    cursor.execute("SELECT * FROM users WHERE name=%s", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({"status": "error", "message": "Username already exists!"})

    # Hash password & convert to string
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Insert user into database
    cursor.execute("INSERT INTO users (name, password, timestamp) VALUES (%s, %s, %s)", 
                   (username, hashed_password, datetime.now()))
    conn.commit()

    # Store username in session
    session["username"] = username

    # Create response with cookies
    response = make_response(jsonify({"status": "success", "message": "Account created successfully!"}))
    response.set_cookie("username", username, max_age=86400, httponly=True)  # 1-day expiry

    cursor.close()
    conn.close()

    return response

# Handle Login
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE name=%s", (username,))
    user = cursor.fetchone()

    if user and bcrypt.checkpw(password.encode('utf-8'), user[2].encode('utf-8')):  # Verify password
        session["username"] = username  # Store in session
        
        response = make_response(jsonify({"status": "success", "message": "Login successful!"}))
        response.set_cookie("username", username, max_age=86400, httponly=True)  # 1-day expiry cookie
        return response

    return jsonify({"status": "error", "message": "Invalid username or password."})


# Check session status
@app.route('/session-status', methods=['GET'])
def session_status():
    username = session.get('username')
    if username:
        return jsonify({"loggedIn": True, "username": username})
    
    # Try checking cookies if session is not found
    username_cookie = request.cookies.get("username")
    if username_cookie:
        return jsonify({"loggedIn": True, "username": username_cookie})

    return jsonify({"loggedIn": False})

# Handle Logout
@app.route("/logout", methods=["GET"])
def logout():
    session.pop("username", None)  # Remove session
    response = make_response(jsonify({"status": "success", "message": "Logged out successfully!"}))
    response.set_cookie("username", "", expires=0)  # Clear cookie
    return response


# Route for face recognition
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

            results.append({"name": recognized_name, "similarity": float(best_similarity)})

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

# Route for face registration
@app.route('/register-face', methods=['POST'])
def register_face():
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')

        if not name or not image_data:
            return jsonify({"status": "error", "message": "Missing name or image data."})

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

        # Extract the first face
        face = img[int(faces[0][1]):int(faces[0][3]), int(faces[0][0]):int(faces[0][2])]
        face_resized = cv2.resize(face, (160, 160))
        face_normalized = face_resized / 255.0
        face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0).float()

        # Extract embedding for the detected face
        embedding = feature_extractor(face_tensor).detach().numpy().flatten()
        embedding = embedding / np.linalg.norm(embedding)  # Normalize embedding

        # Store the new face embedding and name in the CSV file
        with open(STORED_EMBEDDINGS_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name] + embedding.tolist())

        # Add the new face data to the in-memory lists
        stored_embeddings.append(embedding.tolist())
        names_list.append(name)

        return jsonify({"status": "success", "message": f"Registration successful for {name}."})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})

if _name_ == '_main_':
    app.run(debug=True)
