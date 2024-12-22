from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

# Initialize database
db = SQLAlchemy()

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)  # Ensure this line exists
    password = db.Column(db.String(150), nullable=False)
    def is_authenticated(self):
        return True  # Or implement your logic

    def is_active(self):
        return True  # You can customize this logic (e.g., check if the user is not deactivated)

    def is_anonymous(self):
        return False  # Since this model is for authenticated users only

    def get_id(self):
        return str(self.id) 

# Prediction Model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_filename = db.Column(db.String(150), nullable=False)
    result = db.Column(db.String(50), nullable=False)
    severity = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))
