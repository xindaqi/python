from flask import Flask, request, Response, jsonify, Blueprint
from flask import render_template, redirect, url_for
from flask_paginate import Pagination, get_page_parameter
from datetime import datetime, timedelta


from flask_nav import Nav
from flask_nav.elements import *

from app import app, db, manager
from app.models.databases import Info
import requests, json
# from flask_apscheduler import APScheduler 
import schedule
from flask_cors import CORS
import config
import base64
# from api.facebase_structure.endpoints.test_api import ns as test_namespace
# from api.restplus import api, custom_ui
from flask_bootstrap import Bootstrap
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from jinja2 import Markup, Environment, FileSystemLoader
from pyecharts.globals import CurrentConfig
# from pyecharts.options import DataZoomOpts

CurrentConfig.GLOBAL_ENV = Environment(loader=FileSystemLoader("./templates"))

from pyecharts import options as opts 
from pyecharts.charts import Bar, Line, Grid, Page, Pie, Scatter 


CORS(app, supports_credentials=True)

people_show = Blueprint("people", __name__, url_prefix="/api/display")
# where people is the name of Blueprint

def configure_app(flask_app):
    # clear cache browser
    flask_app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
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
	# blue = Blueprint('test',__name__,url_prefix='/blue')
    faceapi = Blueprint('face', __name__, url_prefix='/api')
    # limiter = Limiter(app, key_func=get_remote_address, default_limits=["2 per day", "10 per hour"])
    # limiter.limit("2/minute")(user_show)
    # api.init_app(faceapi)
	# api.init_app(blue)
    # api.add_namespace(test_namespace)
	# custom_ui()
    app.register_blueprint(faceapi)
    app.register_blueprint(people_show)
	# app.register_blueprint(blue)
    db.init_app(app)
    return app
app = create_app()


def query_data():
    infos = Info.query.all()
    print("infos: {}".format(infos))
    for info in infos:
        print("id: {}\nname: {}\naddress: {}\n".format(info.id, info.name, info.address))
        print("=====\n")

def insert_data():
    people = Info(name="小四", address="广东")
    db.session.add_all([people])
    db.session.commit()


if __name__ == "__main__":
	# scheduler = APScheduler()
	# print("++++++")
	# app.config.from_object(Config())
	# scheduler.init_app(app)
	# scheduler.start()
    # db.drop_all()
    '''create database'''
    # db.create_all()
    '''migrate and upgrade'''
    # manager.run()
    '''clear data from database'''
    # clear_data()

    '''add data in database for test'''
    insert_data()
    '''query data in dababase for test'''
    query_data()


	# initializer_app(app)
    # app.run(host='0.0.0.0', port=8090, debug=True)
    
    # student = Student()
    # student_infos = Student.query.filter(Student.u_id=="001").first()
    # student_infos = Student.query.filter_by(u_id="001").first()
    # print("student informations: {}".format(student_infos.teachers[0].name))
    # print("student informations: {}".format(student_infos.teachers[1].name))
