from app import db 
from sqlalchemy import Column

class Info(db.Model):
    __tablename__ = "info"
    __table_args__ = {"mysql_collate":"utf8_general_ci"}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=False)
    address = db.Column(db.String(100), unique=False)
    def __repr__(self):
        return '<Info%r>'%self.id