from flask import Flask, request, Response, jsonify, Blueprint
from flask import render_template, redirect, url_for
from flask_paginate import Pagination, get_page_parameter

from flask_nav import Nav 
from flask_nav.elements import * 
import config
from app import app, db, manager 
from app.models.databases import Student, Teacher 
import requests, json 
from flask_bootstrap import Bootstrap

school_info = Blueprint("school", __name__, url_prefix="/api/school")

@school_info.route("/")
def home():
    return render_template("home.html")

@school_info.route("/student/show", methods=["GET", "POST"])
def student_info_show():
    student_number_per_page = 3
    student_count = Student.query.count()
    page = request.args.get(get_page_parameter(), type=int, default=1)
    start_page = (page-1)*student_number_per_page
    end_page = start_page + student_number_per_page

    students = Student.query.paginate(page, per_page=10)
    return render_template("students/show.html", students=students.items,
    pagination=students, totalNumber=student_count)

@school_info.route("/student/add", methods=["GET", "POST"])
def student_info_add():
    if request.method == "GET":
        return render_template("students/add.html")
    elif request.method == "POST":
        u_id = request.form.get("u_id")
        name = request.form.get("name")
        grade = request.form.get("grade")
        teachers = request.values.getlist("teacher")
        students_info = Student(u_id=u_id, name=name, grade=grade)
        teacher_selected = [Teacher.query.filter_by(id=int(teacher)).first() for teacher in teachers]
        students_info.teachers = teacher_selected
        print("selecting teachers id: {}".format(teachers))
        print("query teachers object: {}".format(teacher_selected))
        db.session.add(students_info)
        db.session.commit()
        return redirect(url_for("school.student_info_show"))

@school_info.route("/student/delete/<int:id>", methods=["GET", "POST"])
def student_info_delete(id):
    student_info = Student.query.filter_by(id=id).first()
    db.session.delete(student_info)
    db.session.commit()
    return redirect(url_for("school.student_info_show"))

@school_info.route("/student/delete_selected", methods=["GET", "POST"])
def student_info_delete_selected():
    student_selected = request.values.getlist("student_selected")
    print("selected students id: {}".format(student_selected))
    for u_id in student_selected:
        student_info = Student.query.filter_by(id=int(u_id)).first()
        db.session.delete(student_info)
    db.session.commit()
    return redirect(url_for("school.student_info_show"))


@school_info.route("/student/edit/<int:id>", methods=["GET", "POST"])
def student_info_edit(id):
    if request.method == "GET":
        student_info = Student.query.filter_by(id=id).first()
        return render_template("students/edit.html", students=student_info)
    elif request.method == "POST":
        student_info = Student.query.filter_by(id=id).first()
        student_info.u_id = request.form.get("u_id")
        student_info.name = request.form.get("name")
        student_info.grade = request.form.get("grade")
        teachers = request.values.getlist("teacher")
        # students_info = Student(u_id=u_id, name=name, grade=grade)
        teacher_selected = [Teacher.query.filter_by(id=int(teacher)).first() for teacher in teachers]
        student_info.teachers = teacher_selected
        print("selecting teachers id: {}".format(teachers))
        print("query teachers object: {}".format(teacher_selected))
        db.session.commit()
        return redirect(url_for("school.student_info_show"))

def configure_app(flask_app):
    flask_app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
    flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = config.SQLALCHEMY_TRACK_MODIFICATIONS 
    flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'# list,none,full
    flask_app.config['RESTPLUS_VALIDATE'] = True
    flask_app.config['RESTPLUS_MASK_SWAGGER'] = False
    flask_app.config['SWAGGER_UI_OPERATION_ID'] = True# right text
    flask_app.config['SWAGGER_UI_REQUEST_DURATION'] = True
    flask_app.config['ERROR_404_HELP'] = False

def create_app():
    app = Flask(__name__)
    bootstrap = Bootstrap(app)
    configure_app(app)
    app.register_blueprint(school_info)
    db.init_app(app)
    return app

app = create_app()

if __name__ == "__main__":
    # db.drop_all()
    db.create_all()
    # manager.run()
    # stu1 = Student(u_id="001", name="小一", grade="100")
    # stu2 = Student(u_id="002", name="小二", grade="100")
    # stu3 = Student(u_id="003", name="小三", grade="100")

    # teach1 = Teacher(u_id="001", name="师一", office="001")
    # teach2 = Teacher(u_id="002", name="师二", office="002")
    # teach3 = Teacher(u_id="003", name="师三", office="003")

    # stu1.teachers = [teach1, teach2]
    # stu2.teachers = [teach2]
    # stu3.teachers = [teach1, teach2, teach3]

    # db.session.add_all([stu1, stu2, stu2])
    # db.session.add_all([teach1, teach2, teach3])

    # db.session.commit()
    # student = Student()
    # student_infos = Student.query.filter(Student.u_id=="001").first()
    # student_infos = Student.query.filter_by(u_id="001").first()
    # print("student informations: {}".format(student_infos.teachers[0].name))
    # print("student informations: {}".format(student_infos.teachers[1].name))

    app.run(host="0.0.0.0", port=8091, debug=True)
