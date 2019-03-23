#-*-coding:utf-8-*-
from app import db
from flask_login import UserMixin
class User(db.Model):
	__tablename__ = 'users'
	id = db.Column(db.Integer, primary_key=True)
	username = db.Column(db.String(80))
	email = db.Column(db.String(320),unique=True)
	password = db.Column(db.String(320),nullable=False)
	phoneNum = db.Column(db.Integer)
	def is_authenticated(self):
		return True

	def is_active(self):
		return True

	def is_anonymous(self):
		return False

	def get_id(self):
		return self.id

	def __repr__(self):
		return "User {}".format(self.username)

class Project(db.Model):
	__tablename__ = 'projects'
	id = db.Column(db.Integer, primary_key=True)
	projectName = db.Column(db.String(320))
	robotId = db.Column(db.Integer, db.ForeignKey('robots.id'))

	def __repr__(self):
		return '<Project %r>' %self.projectName

class Robot(db.Model):
	__tablename__ = 'robots'
	id = db.Column(db.Integer, primary_key=True)
	robotFlag = db.Column(db.Integer,unique=True)
	robotName = db.Column(db.String(320),unique=True)
	projects = db.relationship('Project', backref='robot', lazy='dynamic')

	def __repr__(self):
		return '<Robot %d>' %self.robotFlag



class Data(db.Model):
	__tablename__ = "datas"
	id = db.Column(db.Integer, primary_key=True)
	smallInteger = db.Column(db.SmallInteger)
	bigInteger = db.Column(db.BigInteger)
	floatData = db.Column(db.Float(10))
	numericData = db.Column(db.Numeric(10))
	stringData = db.Column(db.String(250))
	textData = db.Column(db.Text(200))
	mediumText = db.Column(db.Text(65536))
	longText = db.Column(db.Text(16777216))
	largeBinary = db.Column(db.LargeBinary(300))
	mediumBlob = db.Column(db.LargeBinary(65536))
	longBlob = db.Column(db.LargeBinary(16777216))
	pickle = db.Column(db.PickleType)
	mediumPickle = db.Column(db.PickleType(65536))
	longPickle = db.Column(db.PickleType(16777216))
	unicodeData = db.Column(db.Unicode(10))
	unicodeText = db.Column(db.UnicodeText)
	booleanData = db.Column(db.Boolean(0))
	dateData = db.Column(db.Date)
	timeData = db.Column(db.Time)
	dateTime = db.Column(db.DateTime)
	interval = db.Column(db.Interval)
	enumData = db.Column(db.Enum('father', 'mother'))
	def __repr__(self):
		return "Data {}".format(self.id)

















