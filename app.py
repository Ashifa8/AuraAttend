from flask import Flask, render_template, request, redirect, url_for, jsonify, session, make_response
from flask_cors import CORS
import mysql.connector
from datetime import datetime
import bcrypt
import os
import base64
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
from PIL import Image
from io import BytesIO
import pandas as pd
import csv
from werkzeug.utils import secure_filename
from files_embeddings import extract_embeddings  
from bs4 import BeautifulSoup

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = "your_secret_key"


def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="user_database"
    )
# Load MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the feature extractor (InceptionResnetV1)
feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()

# Path to embeddings CSV file
STORED_EMBEDDINGS_PATH = "cleaned_face_embeddings.csv"

# Initialize lists
stored_embeddings = []
names_list = []
class_list = []

# Load stored embeddings and names from the CSV file
if os.path.exists(STORED_EMBEDDINGS_PATH):
    df = pd.read_csv(STORED_EMBEDDINGS_PATH)
    names_list = df['Person'].tolist()
    class_list = df['Class'].tolist()
    stored_embeddings = df.iloc[:, 2:].astype(np.float32).values.tolist()

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Recognition route
@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"status": "error", "message": "No image data received."})

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Error decoding the image."})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces, _ = mtcnn.detect(img_rgb)

        if faces is None or len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected."})

        results = []

        similarity_threshold = 0.5 # 🔄 UPDATED THRESHOLD

        for (x1, y1, x2, y2) in faces:
            face = img[int(y1):int(y2), int(x1):int(x2)]
            face_resized = cv2.resize(face, (160, 160))
            face_normalized = face_resized / 255.0
            face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0).float()

            embedding = feature_extractor(face_tensor).detach().numpy().flatten()

            if embedding.shape[0] == 513:
                embedding = embedding[:512]
            elif embedding.shape[0] != 512:
                return jsonify({"status": "error", "message": "Invalid embedding shape."})

            embedding = embedding / np.linalg.norm(embedding)

            best_similarity = -1
            recognized_name = "Unknown"
            recognized_class = "Unknown"

            for stored_embedding, name, class_name in zip(stored_embeddings, names_list, class_list):
                similarity = cosine_similarity(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    recognized_name = name
                    recognized_class = class_name

            if best_similarity < similarity_threshold:
                recognized_name = "Unknown"
                recognized_class = "Unknown"

            results.append({
                "name": recognized_name,
                "class": recognized_class,
                "similarity": float(best_similarity),
                "threshold": similarity_threshold
            })

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})
    

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.get_json()
        name = data.get('name')
        class_name = data.get('class')
        date_str = datetime.now().strftime("%Y-%m-%d")

        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'message': 'User not logged in'})

        if not name or not class_name:
            return jsonify({'success': False, 'message': 'Missing name or class'})

        # ✅ Query Excel file path from database using user_id and file_name (class_name)
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="user_database"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT file_path FROM excel_files WHERE user_id = %s AND file_name LIKE %s", 
                       (user_id, f"%{class_name}%"))
        result = cursor.fetchone()
        conn.close()

        if not result:
            return jsonify({'success': False, 'message': f"No Excel file found for class '{class_name}'"})

        matching_file = result[0]  # Full path like 'excel_sheets/BSSE_SS1_8th.xlsx'
        print(f"📂 Found Excel file: {matching_file}")

        df = pd.read_excel(matching_file, engine='openpyxl')

        if 'Name' not in df.columns:
            return jsonify({'success': False, 'message': "Excel must contain a 'Name' column."})

        # Add date column if not present
        if date_str not in df.columns:
            df[date_str] = ""

        if name in df['Name'].values:
            df.loc[df['Name'] == name, date_str] = "P"
        else:
            new_row = {col: "" for col in df.columns}
            new_row['Name'] = name
            new_row[date_str] = "P"
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_excel(matching_file, index=False, engine='openpyxl')
        return jsonify({'success': True, 'message': f"Attendance marked for {name} in '{class_name}'."})

    except Exception as e:
        return jsonify({'success': False, 'message': f"Error updating Excel: {str(e)}"})


@app.route('/get_class_name_for_recognized_face', methods=['POST', 'OPTIONS'])
def get_class_name_for_recognized_face():
    # Handle OPTIONS request (pre-flight CORS request)
    if request.method == 'OPTIONS':
        return '', 200  # Respond with a 200 OK to pre-flight requests

    try:
        data = request.get_json()
        name = data.get('name')

        if not name:
            return jsonify({"success": False, "message": "Name is required"})

        # Path to your cleaned embeddings CSV (adjust path if necessary)
        embeddings_file_path = 'cleaned_face_embeddings.csv'

        # Check if the file exists
        if not os.path.exists(embeddings_file_path):
            return jsonify({"success": False, "message": f"Embeddings file '{embeddings_file_path}' not found."})

        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(embeddings_file_path)

        # Strip any extra spaces from column names (if any)
        df.columns = df.columns.str.strip()

        # Check if 'Class' and 'Person' columns exist (with capital 'C' and 'P')
        if 'Class' not in df.columns or 'Person' not in df.columns:
            raise ValueError("CSV must contain 'Class' and 'Person' columns.")

        # Find the row where the 'Person' matches the recognized name
        matched_row = df[df['Person'] == name]

        if matched_row.empty:
            return jsonify({"success": False, "message": "Name not found in the CSV."})

        # Extract the class name from the matched row
        class_name = matched_row.iloc[0]['Class']

        return jsonify({"success": True, "className": class_name})

    except ValueError as ve:
        # Handle specific errors like missing columns in CSV
        return jsonify({"success": False, "message": str(ve)})

    except Exception as e:
        # General error handling
        print(f"❌ Error in get_class_name_for_recognized_face: {str(e)}")
        return jsonify({"success": False, "message": f"An error occurred: {str(e)}"})

@app.route('/get_user_id', methods=['GET'])
def get_user_id():
    user_id = session.get('user_id')
    username = session.get('username')
    if user_id:
        return jsonify({"status": "success", "user_id": user_id, "username": username})
    else:
        return jsonify({"status": "error", "message": "User not logged in"}), 401
@app.route('/get_session_data', methods=['GET'])
def get_session_data():
    return jsonify({
        "class": session.get('current_class'),
        "excel_path": session.get('excel_path'),
        "user_id": session.get('user_id'),
        "username": session.get('username')
    })


@app.route("/create_excel", methods=["POST"])
def create_excel():
    data = request.get_json()
    sheet_name = data["sheet_name"]
    user_id = session.get("user_id")

    if not user_id:
        return jsonify(success=False, message="Not logged in")

    filename = f"{sheet_name}.xlsx"
    UPLOAD_FOLDER = "excel_sheets"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    df = pd.DataFrame(columns=["Name", "Status"])
    df.to_excel(file_path, index=False)

    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO excel_files (user_id, file_name, file_path, created_at)
        VALUES (%s, %s, %s, NOW())
    """, (user_id, filename, file_path))
    db.commit()
    excel_id = cursor.lastrowid

    luckysheet_data = [{
        "name": "Sheet1",
        "color": "",
        "status": 1,
        "order": 0,
        "row": 20,
        "column": 10,
        "celldata": []
    }]

    return jsonify(success=True, sheet_data=luckysheet_data, excel_id=excel_id)

@app.route("/get_excels", methods=["GET"])
def get_excels():
    user_id = session.get("user_id")
    if not user_id:
        return jsonify(success=False, message="Not logged in")

    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("""
        SELECT id, file_name, file_path, created_at
        FROM excel_files
        WHERE user_id = %s
        ORDER BY created_at DESC
    """, (user_id,))
    records = cursor.fetchall()

    excel_files = [{
        "id": row[0],
        "file_name": row[1],
        "file_path": row[2],
        "created_at": row[3].strftime("%Y-%m-%d %H:%M:%S")
    } for row in records]

    return jsonify(success=True, excels=excel_files)

@app.route("/load_excel/<int:excel_id>", methods=["GET"])
def load_excel(excel_id):
    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT file_path FROM excel_files WHERE id = %s", (excel_id,))
    result = cursor.fetchone()

    if not result:
        return jsonify(success=False, message="Excel not found")
    
    file_path = result[0]
    if not os.path.exists(file_path):
        return jsonify(success=False, message="File missing")

    df = pd.read_excel(file_path, header=None)

    celldata = []
    for r, row in df.iterrows():
        for c, value in enumerate(row):
            celldata.append({
                "r": r,
                "c": c,
                "v": { "v": str(value) }
            })

    sheet_data = [{
        "name": "Sheet1",
        "status": 1,
        "order": 0,
        "row": max(20, len(df)+10),
        "column": max(10, len(df.columns)+5),
        "celldata": celldata
    }]

    return jsonify(success=True, sheet_data=sheet_data)

@app.route("/save_excel", methods=["POST"])
def save_excel():
    data = request.get_json()
    sheet_json = data.get("sheet_json")
    excel_id = data.get("excel_id")

    if not sheet_json or not isinstance(sheet_json, list):
        return jsonify(success=False, message="Invalid sheet format")

    celldata = sheet_json[0].get("celldata", [])
    if not celldata:
        return jsonify(success=False, message="No data to save")

    max_r = max(cell["r"] for cell in celldata)
    max_c = max(cell["c"] for cell in celldata)
    table = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]

    for cell in celldata:
        r, c, v = cell["r"], cell["c"], cell["v"].get("v", "")
        table[r][c] = v

    df = pd.DataFrame(table)

    db = get_db_connection()
    cursor = db.cursor()
    cursor.execute("SELECT file_path FROM excel_files WHERE id = %s", (excel_id,))
    result = cursor.fetchone()

    if not result:
        return jsonify(success=False, message="Excel file not found")

    path = result[0]
    df.to_excel(path, index=False, header=False)

    return jsonify(success=True)

@app.route("/delete_excel/<int:excel_id>", methods=["POST"])
def delete_excel(excel_id):
    db = get_db_connection()
    cursor = db.cursor()

    # Fetch the file path for the specified Excel ID
    cursor.execute("SELECT file_path FROM excel_files WHERE id = %s", (excel_id,))
    result = cursor.fetchone()

    if not result:
        return jsonify(success=False, message="Excel file not found")

    file_path = result[0]
    print(f"Attempting to delete file at: {file_path}")  # Debugging line

    # Delete the file from the file system
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        return jsonify(success=False, message="File does not exist on the server")

    # Remove the record from the database
    cursor.execute("DELETE FROM excel_files WHERE id = %s", (excel_id,))
    db.commit()

    return jsonify(success=True, message="Excel file deleted successfully")


UPLOAD_FOLDER = 'uploads'  # Base upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------- Check if CSV exists, write header if not -------------------- #
csv_file = 'cleaned_face_embeddings.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Person'] + [f'Embedding_{i+1}' for i in range(512)])

@app.route('/upload_files', methods=['POST'])
def upload_files():
    try:
        # Check if user is logged in
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': 'User not logged in'})
        
        user_id = session['user_id']
        folder_name = request.form.get('folder_name')

        if not folder_name:
            return jsonify({'success': False, 'error': 'Folder name missing'})

        # Connect to DB
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='user_database'
        )
        cursor = conn.cursor()

        # Get folder ID
        cursor.execute("SELECT id FROM folders WHERE folder_name = %s AND user_id = %s", (folder_name, user_id))
        folder_row = cursor.fetchone()
        if not folder_row:
            return jsonify({'success': False, 'error': 'Folder not found in database'})

        folder_id = folder_row[0]

        # Create user folder path if not exists
        user_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id), folder_name)
        os.makedirs(user_folder_path, exist_ok=True)

        uploaded_files = request.files.getlist('files')
        if not uploaded_files:
            return jsonify({'success': False, 'error': 'No files received'})

        for file in uploaded_files:
            if file.filename == '':
                continue

            filename = secure_filename(file.filename)
            file_path = os.path.join(user_folder_path, filename)
            file.save(file_path)
            print(f"File saved to {file_path}")

            # Store in database
            cursor.execute(
                "INSERT INTO files (folder_id, file_name, file_path, uploaded_at) VALUES (%s, %s, %s, %s)",
                (folder_id, filename, file_path, datetime.now())
            )

            # Automatically calculate and save embeddings after the file is uploaded
            try:
                embeddings = extract_embeddings(file_path)  # Use your existing embedding extraction function
                if embeddings:
                    # Save the embeddings to the CSV file
                    with open('cleaned_face_embeddings.csv', mode='a', newline='') as f:
                        writer = csv.writer(f)
                        for emb in embeddings:
                            writer.writerow([folder_name, os.path.splitext(filename)[0]] + emb.tolist())
                    print(f"Embeddings saved for {filename}")
                else:
                    return jsonify({'success': False, 'error': 'Failed to extract embeddings from image'})

            except Exception as e:
                return jsonify({'success': False, 'error': f'Error extracting embeddings: {str(e)}'})

        # Commit to DB and close the connection
        conn.commit()
        cursor.close()
        conn.close()

        print("Files uploaded and embeddings calculated successfully.")
        return jsonify({'success': True, 'message': 'Files uploaded, stored in database, and embeddings calculated'})

    except mysql.connector.Error as e:
        print(f"MySQL error: {str(e)}")
        return jsonify({'success': False, 'error': f'MySQL error: {str(e)}'})
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'success': False, 'error': f'Unexpected error: {str(e)}'})

@app.route('/get_files/<folder_name>', methods=['GET'])
def get_files(folder_name):
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "error": "User not logged in"})

    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='user_database'
        )
        cursor = conn.cursor()

        # Get folder ID based on folder name and user
        cursor.execute("SELECT id FROM folders WHERE folder_name = %s AND user_id = %s", (folder_name, user_id))
        folder_result = cursor.fetchone()

        if folder_result:
            folder_id = folder_result[0]

            # Fetch files belonging to the folder
            cursor.execute("SELECT id, file_name, file_path FROM files WHERE folder_id = %s", (folder_id,))
            files = cursor.fetchall()

            # Prepare the file list with id, file_name, file_path
            file_list = [{"id": file[0], "file_name": file[1], "file_path": file[2]} for file in files]

            return jsonify({"success": True, "files": file_list})
        else:
            return jsonify({"success": True, "files": []})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/uploads/files/<filename>')
def uploaded_file(filename):
    print(f"Serving file: {filename}")  # Print statement to debug
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "error": "User not logged in"})

    # Construct the path to the user's specific folder
    user_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    file_path = os.path.join(user_folder_path, filename)

    # Make sure the file exists
    if os.path.exists(file_path):
        return send_from_directory(user_folder_path, filename)
    else:
        return jsonify({"success": False, "error": "File not found"})

@app.route('/delete_file', methods=['POST'])
def delete_file():
    try:
        file_id = request.form.get('file_id')
        if not file_id:
            return jsonify({"success": False, "error": "No file_id provided."})

        user_id = session.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "User not logged in"})

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='',
            database='user_database'
        )
        cursor = conn.cursor()

        # Get file path before deletion
        cursor.execute("SELECT file_path FROM files WHERE id = %s", (file_id,))
        file_row = cursor.fetchone()

        if not file_row:
            return jsonify({"success": False, "error": "File not found in database"})

        file_path = file_row[0]

        # Delete file from disk
        if os.path.exists(file_path):
            os.remove(file_path)

        # Delete file record from DB
        cursor.execute("DELETE FROM files WHERE id = %s", (file_id,))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"success": True, "message": "File deleted successfully."})

    except Exception as e:
        print(f"Error deleting file: {e}")
        return jsonify({"success": False, "error": str(e)})

    
@app.route('/delete_folder', methods=['POST'])
def delete_folder():
    data = request.get_json()
    folder_name = data.get('folder_name')

    if not folder_name:
        return jsonify({'success': False, 'error': 'Folder name not provided'})

    try:
        db = get_db_connection()
        cursor = db.cursor()

        cursor.execute("DELETE FROM folders WHERE folder_name = %s", (folder_name,))
        db.commit()

        if cursor.rowcount > 0:
            return jsonify({'success': True, 'message': 'Folder deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Folder not found in the database'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

    finally:
        if 'cursor' in locals(): cursor.close()
        if 'db' in locals(): db.close()



@app.route('/create_folder', methods=['POST'])
def create_folder():
    # Step 1: Check if user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'User not logged in'}), 401

    # Step 2: Get folder name from the request
    data = request.get_json()
    folder_name = data.get('folder_name')

    if not folder_name:
        return jsonify({'success': False, 'error': 'Folder name is required'}), 400

    try:
        # Step 3: Create full folder path: uploads/user_id/folder_name
        folder_path = os.path.join('uploads', str(user_id), folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Step 4: Insert folder record into database
        conn = get_db_connection()
        cursor = conn.cursor()

        created_at = datetime.now()
        cursor.execute('''INSERT INTO folders (user_id, folder_name, created_at, folder_path) 
                          VALUES (%s, %s, %s, %s)''', 
                       (user_id, folder_name, created_at, folder_path))

        conn.commit()
        cursor.close()
        conn.close()

        # Step 5: Return success response
        return jsonify({
            'success': True,
            'message': 'Folder created successfully',
            'folder_name': folder_name
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Database or server error: {str(e)}"
        }), 500

@app.route('/get_folders', methods=['GET'])
def get_folders():
    # Step 1: Check if user is logged in
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'message': 'User not logged in'}), 401

    # Step 2: Fetch the user's folders from the database
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('SELECT folder_name FROM folders WHERE user_id = %s', (user_id,))
    folders = cursor.fetchall()

    cursor.close()
    conn.close()

    # If no folders found, return an empty list instead of error
    return jsonify({'success': True, 'folders': folders}), 200

    

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, password FROM users WHERE name=%s", (username,))
    user = cursor.fetchone()
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        session["user_id"] = user['id']
        session["username"] = username
        response = make_response(jsonify({"status": "success", "message": "Login successful!"}))
        response.set_cookie("username", username, max_age=86400, httponly=True)
        return response
    return jsonify({"status": "error", "message": "Invalid username or password."})

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    password = data['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE name=%s", (username,))
    existing_user = cursor.fetchone()
    if existing_user:
        return jsonify({"status": "error", "message": "Username already exists!"})
    
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    cursor.execute("INSERT INTO users (name, password, timestamp) VALUES (%s, %s, %s)", 
                   (username, hashed_password, datetime.now()))
    conn.commit()
    user_id = cursor.lastrowid
    session["user_id"] = user_id
    session["username"] = username
    response = make_response(jsonify({"status": "success", "message": "Account created successfully!"}))
    response.set_cookie("username", username, max_age=86400, httponly=True)
    cursor.close()
    conn.close()
    return response

@app.route('/session-status', methods=['GET'])
def session_status():
    username = session.get('username')
    if username:
        return jsonify({"loggedIn": True, "username": username})
    username_cookie = request.cookies.get("username")
    if username_cookie:
        return jsonify({"loggedIn": True, "username": username_cookie})
    return jsonify({"loggedIn": False})

@app.route("/logout", methods=["GET"])
def logout():
    session.pop("username", None)
    session.pop("user_id", None)
    response = make_response(jsonify({"status": "success", "message": "Logged out successfully!"}))
    response.set_cookie("username", "", expires=0)
    return response


# Serve the reviews page
@app.route('/reviews')
def reviews():
    return render_template('reviews.html')  # This will render the reviews.html from the templates folder

# API to get reviews from the database
@app.route('/get_reviews', methods=['GET'])
def get_reviews():
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        cursor.execute("SELECT * FROM reviews")
        reviews = cursor.fetchall()

        cursor.close()
        db.close()

        return jsonify({"reviews": reviews})

    except Exception as e:
        return jsonify({"error": str(e)})
    
@app.route('/submit_review', methods=['POST'])
def submit_review():
    try:
        # Get JSON data from the request
        data = request.json
        user_name = data.get('user_name', '').strip()
        thoughts = data.get('thoughts', '').strip()
        rating = data.get('rating', '').strip()

        # Validate inputs
        if not user_name or not thoughts or not rating:
            return jsonify({"message": "All fields are required!"}), 400

        try:
            rating = int(rating)
            if rating < 1 or rating > 5:
                return jsonify({"message": "Rating must be between 1 and 5!"}), 400
        except ValueError:
            return jsonify({"message": "Invalid rating value!"}), 400

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Connect to database and insert the review
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO reviews (user_name, thoughts, rating, created_at) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (user_name, thoughts, rating, timestamp))
        conn.commit()
        
        # Close database connection
        cursor.close()
        conn.close()

        return jsonify({"message": "Review submitted successfully!"})

    except mysql.connector.Error as err:
        return jsonify({"message": f"Database error: {str(err)}"}), 500
    except Exception as e:
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500
    


# Load MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Load the feature extractor (InceptionResnetV1)
feature_extractor = InceptionResnetV1(pretrained='vggface2').eval()

# Path to embeddings CSV file
STORED_EMBEDDINGS_PATH = "cleaned_face_embeddings.csv"

# Initialize lists
stored_embeddings = []
names_list = []
class_list = []

# Load stored embeddings and names from the CSV file
if os.path.exists(STORED_EMBEDDINGS_PATH):
    df = pd.read_csv(STORED_EMBEDDINGS_PATH)
    names_list = df['Person'].tolist()
    class_list = df['Class'].tolist()
    stored_embeddings = df.iloc[:, 2:].astype(np.float32).values.tolist()

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Recognition route
@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        image_data = data.get('image')

        if not image_data:
            return jsonify({"status": "error", "message": "No image data received."})

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"status": "error", "message": "Error decoding the image."})

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces, _ = mtcnn.detect(img_rgb)

        if faces is None or len(faces) == 0:
            return jsonify({"status": "error", "message": "No face detected."})

        results = []

        similarity_threshold = 0.5 # 🔄 UPDATED THRESHOLD

        for (x1, y1, x2, y2) in faces:
            face = img[int(y1):int(y2), int(x1):int(x2)]
            face_resized = cv2.resize(face, (160, 160))
            face_normalized = face_resized / 255.0
            face_tensor = torch.tensor(face_normalized).permute(2, 0, 1).unsqueeze(0).float()

            embedding = feature_extractor(face_tensor).detach().numpy().flatten()

            if embedding.shape[0] == 513:
                embedding = embedding[:512]
            elif embedding.shape[0] != 512:
                return jsonify({"status": "error", "message": "Invalid embedding shape."})

            embedding = embedding / np.linalg.norm(embedding)

            best_similarity = -1
            recognized_name = "Unknown"
            recognized_class = "Unknown"

            for stored_embedding, name, class_name in zip(stored_embeddings, names_list, class_list):
                similarity = cosine_similarity(embedding, stored_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    recognized_name = name
                    recognized_class = class_name

            if best_similarity < similarity_threshold:
                recognized_name = "Unknown"
                recognized_class = "Unknown"

            results.append({
                "name": recognized_name,
                "class": recognized_class,
                "similarity": float(best_similarity),
                "threshold": similarity_threshold
            })

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": f"Error: {str(e)}"})


@app.route('/register-face', methods=['POST'])
def register_face():
    try:
        data = request.get_json()
        name = data['name']
        class_name = data['class']
        image_data = data['image']

        # Decode and load the image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(image)

        # Detect face using MTCNN
        boxes, _ = mtcnn.detect(img_np)
        if boxes is None or len(boxes) == 0:
            return jsonify({"status": "error", "message": "No face detected during registration."})

        x1, y1, x2, y2 = [int(b) for b in boxes[0]]
        face = img_np[y1:y2, x1:x2]

        # Resize the face
        face_resized = cv2.resize(face, (160, 160))

        # Augmentations for better generalization
        augmentations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((160, 160), scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])

        embeddings_list = []

        for _ in range(5):  # Generate 5 augmented embeddings
            augmented_face = augmentations(face_resized)
            face_tensor = augmented_face.unsqueeze(0).float()
            embedding = feature_extractor(face_tensor).detach().numpy().flatten()

            if embedding.shape[0] == 513:
                embedding = embedding[:512]
            elif embedding.shape[0] != 512:
                continue  # Skip invalid shapes

            embeddings_list.append(embedding)

        # Save to CSV
        csv_file = STORED_EMBEDDINGS_PATH
        if not os.path.isfile(csv_file):
            header = ['Class', 'Person'] + [f'embedding{i+1}' for i in range(512)]
            df = pd.DataFrame(columns=header)
            df.to_csv(csv_file, index=False)

        rows = [[class_name, name] + emb.tolist() for emb in embeddings_list]
        df_new = pd.DataFrame(rows)
        df_new.to_csv(csv_file, mode='a', header=False, index=False)

        # Reload embeddings in memory
        df = pd.read_csv(STORED_EMBEDDINGS_PATH)
        names_list.clear()
        class_list.clear()
        stored_embeddings.clear()
        names_list.extend(df['Person'].tolist())
        class_list.extend(df['Class'].tolist())
        stored_embeddings.extend(df.iloc[:, 2:].astype(np.float32).values.tolist())

        return jsonify({'status': 'success', 'message': f'{name} from {class_name} registered successfully.'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
