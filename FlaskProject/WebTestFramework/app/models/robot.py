from flask import Blueprint,render_template,request
from app import db

from .databases import Robot

robot = Blueprint('robot', __name__, url_prefix='/robot')

@robot.route('/add/',methods=['POST', 'GET'])
def add():
	if request.method == 'POST':
		robotFlag = request.form.get('robotFlag',None)
		robotName = request.form.get('robotName', None)
		if not robotFlag:
			return 'Try again!'
		robotInfo = Robot(robotFlag=robotFlag, robotName=robotName)
		db.session.add(robotInfo)
		db.session.commit()
		robots = Robot.query.all()
		return render_template('robot/show.html',robots=robots)
	# robots可迭代，需迭代取值
	robots = Robot.query.all()
	return render_template('robot/add.html',robots=robots)

@robot.route('/query/', methods=['GET', 'POST'])
def query():
	if request.method == 'POST':
		robotFlag = request.form.get('robotFlag',None)
		print(robotFlag)
		if not robotFlag:
			return render_template('robot/show.html')

	robotQuery = Robot.query.filter_by(robotFlag=robotFlag).first()
	# robotQuery不可迭代，直接取值
	print(robotQuery.robotName)
	print(robotQuery.robotFlag)
	return render_template('robot/show.html',robotQuery=robotQuery)


@robot.route('/show/',methods=['GET', 'POST'])
def show():
	robots = Robot.query.all()
	return render_template('robot/show.html',robots=robots)