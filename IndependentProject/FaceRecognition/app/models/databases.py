from app import db 
from sqlalchemy import Column

class FaceString(db.Model):
	__tablename__ = "face_strings"
	__table_args__ = {'mysql_collate':'utf8_general_ci'}
	id = db.Column(db.Integer, primary_key=True)
	face_feature = db.Column(db.String(361), unique=True)
	name = db.Column(db.String(361), unique=False)
	sex = db.Column(db.String(361), unique=False)
	address = db.Column(db.String(361), unique=False)
	project = db.Column(db.String(361), unique=False)
	position = db.Column(db.String(361), unique=False)

class FaceFloat(db.Model):
	__tablename__ = "face_floats"
	__table_args__ = {'mysql_collate':'utf8_general_ci'}
	id = db.Column(db.Integer, primary_key=True)
	face_feature = db.Column(db.Float, unique=True)
	name = db.Column(db.String(361), unique=False)
	sex = db.Column(db.String(361), unique=False)
	address = db.Column(db.String(361), unique=False)
	project = db.Column(db.String(361), unique=False)
	position = db.Column(db.String(361), unique=False)

class FaceText(db.Model):
	__tablename__ = "face_texts"
	__table_args__ = {'mysql_collate':'utf8_general_ci'}
	id = db.Column(db.Integer, primary_key=True)
	face_feature = db.Column(db.Text(65536))
	name = db.Column(db.String(361), unique=False)
	sex = db.Column(db.String(361), unique=False)
	address = db.Column(db.String(361), unique=False)
	project = db.Column(db.String(361), unique=False)
	position = db.Column(db.String(361), unique=False)



