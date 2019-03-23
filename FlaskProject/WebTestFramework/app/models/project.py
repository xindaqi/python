from flask import Blueprint,render_template,request
from app import db
from .databases import Project
from xlrd import open_workbook
import os
from flask_paginate import Pagination, get_page_parameter

basedir = os.path.abspath(os.path.dirname(__name__))

project = Blueprint('project', __name__, url_prefix='/project')

@project.route('/add/', methods=['GET', 'POST'])
def add():
	if request.method == 'POST':
		prjName = request.form.get('prjName',None)
		robotId = request.form.get('robotId',None)
		prjInfo = Project(projectName=prjName, robotId=robotId)
		print(type(prjInfo))
		db.session.add(prjInfo)
		db.session.commit()
		projects = Project.query.all()
		return render_template('project/show.html', projects=projects)
	projects = Project.query.all()
	return render_template('project/add.html',projects=projects)

# 从excel导入项目
@project.route('/addexcel/', methods=['GET', 'POST'])
def addexcel():
	if request.method == 'POST':
		ret = {'status':False, 'data':None, 'error':None}
		uploadfile = request.files.get("excelFile")
		if not uploadfile:
			return 'no files upload'

		destination = open(os.path.join(basedir, uploadfile.filename), 'wb+')
		print(destination)

		path = basedir + "/static"
		print(path)
		if not os.path.exists(path):
			os.mkdir(path)

		file_path = path + uploadfile.filename
		print(file_path)
		print(uploadfile.filename)
		uploadfile.save(file_path)

		excelName = file_path

		bk = open_workbook(excelName, encoding_override="utf-8")
		sh = bk.sheets()[0]
		nrows = sh.nrows
		ncols = sh.ncols

		row_data = sh.row_values(1)
		for i in range(1, nrows):
			print(i)


		try:
			sh = bk.sheets()[0]
		except:
			print("no sheet in %s named sheet1" %excelName)
		else:
			nrows = sh.nrows
			ncols = sh.ncols
			row_list = []
			for i in range(1, nrows):
				row_data = sh.row_values(i)
				project = Project()
				project.projectName = row_data[0]
				project.robotId = int(row_data[1])
				db.session.add(project)
				db.session.commit()
		projects = Project.query.all()
		return render_template('project/show.html', projects=projects)


@project.route('/delete/<int:idd>',methods=['GET', 'POST'])
def delete(idd):
	# idd = request.args.get('id')
	# print(idd)
	# return str(idd)
	prj = Project.query.filter_by(id=idd).first()
	# return prj.projectName
	db.session.delete(prj)
	db.session.commit()
	projects = Project.query.all()
	return render_template('project/show.html',projects=projects)

@project.route('/edit/<int:idd>', methods=['GET', 'POST'])
def edit(idd):
	if request.method == "POST":
		prj = Project.query.filter_by(id=idd).first()
		prj.projectName = request.form.get('prjName', None)
		prj.robotId = request.form.get('robotId', None)
		# prjInfo = Project(projectName=prjName, robotId=robotId)
		# db.session.add(prjInfo)
		db.session.commit()
		projects = Project.query.all()
		return render_template('project/show.html',projects=projects)
	prj = Project.query.filter_by(id=idd).first()
	return render_template('project/edit.html',prj=prj)

@project.route('/show/', methods=['GET', 'POST'])
def show():

	# 每页显示项目数量
	perPage = 3
	# 数据库项目总量
	totalNumber = Project.query.count()
	# print(total)
	# 获取页码，默认第一页
	page = request.args.get(get_page_parameter(),type=int, default=1)
	print(page)
	# 每页起始位置
	start = (page-1)*perPage
	# 每页结束位置
	end = start + perPage

	projects = Project.query.paginate(page, per_page=10)

	# 项目处理
	# projects = Project.query.slice(start, end)
	# dataNumber = Pagination(bs_version=3, page=page, total=total, serach=True, record_name='projects')

	# context = {
	# "pagination":pagination,
	# "projects":projects
	# }
	# projects = Project.query.filter().order_by(Project.id.desc()).paginate(page=1, per_page=2)
	return render_template('project/show.html', projects=projects.items, pagination=projects,totalNumber=totalNumber)


	# prjName = request.form.get('prjName', None)
	# robotId = request.form.get('robotId', None)
	# prjInfo = Project(projectName=prjName, robotId=robotId)
	# db.session.add(prjInfo)
	# db.session.commit()
	# projects = Project.query.all()
	# return render_template('project/edit.html', projects=projects)
	


	 