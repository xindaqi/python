from app import db
from flask_login import UserMixin

class User(UserMixin, db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(320), unique=False, nullable=True)
    password = db.Column(db.String(320), unique=False, nullable=False)
    def __init__(self):
        self.id = 250
        self.name = "admin"
        self.password = "admin"

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False 

    def get_id(self):
        return self.id