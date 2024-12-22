import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid
from flask_migrate import Migrate
from models import db, User, Prediction  # Import from models.py

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'Nithin@123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///pneumonia_detection.db'
app.config['UPLOAD_FOLDER'] = r'C:\Users\nithi\OneDrive\Desktop\final year project\static\reports'
app.config['REPORTS_FOLDER'] = r'C:\Users\nithi\OneDrive\Desktop\final year project\static\uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max file size 16MB

# Initialize Flask-Migrate
migrate = Migrate(app, db)

# Initialize the database
db.init_app(app)

# Initialize Login Manager
login_manager = LoginManager()
login_manager.init_app(app)

# Load pre-trained model
model = tf.keras.models.load_model(r'C:\Users\nithi\OneDrive\Desktop\final year project\pneumonia_model.keras')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = str(uuid.uuid4()) + '.jpg'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Open and preprocess the image
        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Resize the image to 150x150 (the expected input shape for the model)
        img = img.resize((150, 150))

        # If the image is grayscale (1 channel), convert it to RGB (3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert image to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Add batch dimension to the image (shape becomes (1, 150, 150, 3))
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using the pre-trained model
        prediction = model.predict(img_array)

        # Interpret the model's output
        result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
        severity = prediction[0][0] * 100 if result == 'Pneumonia' else 0

        # Save the prediction to the database
        new_prediction = Prediction(image_filename=filename, result=result, severity=severity, user_id=current_user.id)
        db.session.add(new_prediction)
        db.session.commit()

        return render_template('result.html', result=result, severity=severity, prediction_id=new_prediction.id)

    else:
        flash('Invalid file type. Only .jpg, .png, .jpeg files are allowed.', 'danger')
        return redirect(request.url)

@app.route('/dashboard')
@login_required
def dashboard():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', predictions=predictions)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))

        flash('Login failed. Check your username and/or password.', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash("Passwords don't match.", 'danger')
            return redirect(request.url)

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/download_report/<int:prediction_id>')
@login_required
def download_report(prediction_id):
    prediction = Prediction.query.get_or_404(prediction_id)

    # Create the report content
    report_content = f"Prediction Report:\n\nImage Filename: {prediction.image_filename}\nResult: {prediction.result}\nSeverity: {prediction.severity}%\n"
    report_filename = f"{prediction.id}_report.txt"
    report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)

    # Write the report to a text file
    with open(report_path, 'w') as report_file:
        report_file.write(report_content)

    # Send the report to the user
    return send_from_directory(app.config['REPORTS_FOLDER'], report_filename)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create tables in the database
    app.run(debug=True)
