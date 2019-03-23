from app import db 
from sqlalchemy import Column

class Knowledge(db.Model):
	__tablename__ = "knowledge"
	__table_args__ = {'mysql_collate':'utf8_general_ci'}
	id = db.Column(db.Integer, primary_key=True)
	question = db.Column(db.String(320), unique=True)
	answer = db.Column(db.String(320), unique=True)
	def __repr__(self):
		return "Knowledge:{}".format(self.question)
# class FaceInfo(db.Model):
# 	__tablename__ = "face_info"
# 	__table_args__ = {'mysql_collate':'utf8_general_ci'}
# 	id = db.Column(db.Integer, primary_key=True)
# 	face_feature_value = db.Column(db.String)





















