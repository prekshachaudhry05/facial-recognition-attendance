import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime
from flask_bcrypt import Bcrypt

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

bcrypt = Bcrypt(app)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017')
db = client['face_attendance']

# Existing collections
users_col = db['users']
attendance_col = db['attendance_logs']

# New collection for authentication users
auth_users_col = db['auth_users']


# Load face detection and recognizer
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

DATASET_PATH = './face_dataset'

def train_recognizer():
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for user_folder in os.listdir(DATASET_PATH):
        folder_path = os.path.join(DATASET_PATH, user_folder)
        if not os.path.isdir(folder_path):
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_label)
        label_map[current_label] = user_folder
        current_label += 1

    if faces:
        recognizer.train(faces, np.array(labels))
    return label_map

label_map = train_recognizer()

@app.route('/register', methods=['POST'])
def register():
    if 'name' not in request.form or 'image' not in request.files:
        return jsonify({"error": "Both 'name' and 'image' are required"}), 400

    name = request.form['name'].strip().lower()
    img_file = request.files['image']

    user_folder = os.path.join(DATASET_PATH, name)
    os.makedirs(user_folder, exist_ok=True)

    img_path = os.path.join(user_folder, f'{len(os.listdir(user_folder)) + 1}.jpg')
    img_file.save(img_path)

    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        os.remove(img_path)
        return jsonify({"error": "No face detected in the uploaded image"}), 400
    (x, y, w, h) = faces[0]
    face_img = gray[y:y + h, x:x + w]
    cv2.imwrite(img_path, face_img)

    existing_user = users_col.find_one({"name": name})
    if not existing_user:
        users_col.insert_one({"name": name})
        print(f"User '{name}' registered in DB.")
    else:
        print(f"User '{name}' already exists, not added.")

    global label_map
    label_map = train_recognizer()
    return jsonify({"message": f"User '{name}' registered successfully."}), 200


@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400
    img_file = request.files['image']
    img_np = np.frombuffer(img_file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    if len(faces) == 0:
        return jsonify({"message": "No face detected"}), 200  # No face, not an error

    (x, y, w, h) = faces[0]
    face_img = gray[y:y + h, x:x + w]

    try:
        label, confidence = recognizer.predict(face_img)
        user_name = label_map.get(label, "Unknown").strip().lower()
        print(f"Recognized: {user_name}")

        user = users_col.find_one({"name": user_name.strip().lower()})
        if user:
            attendance_col.insert_one({
                "user_id": user["_id"],
                "name": user_name,
                "timestamp": datetime.utcnow()
            })

            print(f"Attendance inserted for {user_name}")
        else:
            print(f"No matching user in DB for {user_name}")

        return jsonify({"name": user_name, "confidence": confidence}), 200
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({"error": "Face not recognized"}), 400

@app.route('/logs', methods=['GET'])
def logs():
    logs = list(attendance_col.find().sort("timestamp", -1))
    for log in logs:
        log["_id"] = str(log["_id"])
        log["user_id"] = str(log["user_id"])
        log["timestamp"] = log["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    return jsonify(logs)

@app.route('/admin-signup', methods=['POST'])
def admin_signup():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    if auth_users_col.find_one({'email': email}):
        return jsonify({'error': 'Email already registered'}), 400

    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    auth_users_col.insert_one({
        'email': email,
        'password': pw_hash,
        'is_admin': True
    })
    return jsonify({'message': 'Admin account created'}), 201



@app.route('/login', methods=['POST'])
def login():
    print("Login attempt, headers:", request.headers)
    print("Raw data:", request.get_data())
    try:
        data = request.get_json(force=True)
        print("Parsed JSON:", data)
    except Exception as e:
        print("JSON parsing error:", e)
        return jsonify({'error': 'Invalid JSON'}), 400

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    user = auth_users_col.find_one({'email': email})
    if not user or not bcrypt.check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid email or password'}), 401

    return jsonify({
        'message': 'Login successful',
        'email': user['email'],
        'is_admin': user.get('is_admin', False)
    }), 200


@app.route('/users', methods=['GET'])
def users():
    users_list = list(users_col.find({}, {"_id": 1, "name": 1}))
    for u in users_list:
        u["_id"] = str(u["_id"])
    return jsonify(users_list)

@app.route('/update-user-photo', methods=['POST'])
def update_user_photo():
    user_id = request.form.get('user_id')
    image_file = request.files.get('image')
    if not user_id or not image_file:
        return jsonify({"error": "User ID and photo are required"}), 400
    # Implement file saving and user photo update logic here
    
    # Return success message
    return jsonify({"message": "User photo updated successfully"})


if __name__ == '__main__':
    app.run(debug=True)
