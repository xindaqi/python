from flask import Flask, request, Response, jsonify, Blueprint
from flask import render_template, redirect, url_for
from flask_paginate import Pagination, get_page_parameter
from datetime import datetime, timedelta
import xlrd, xlwt
from argparse import ArgumentParser  
from queue import Queue


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
    configure_app(app)
    db.init_app(app)
    return app
app = create_app()

def insert_data():
    people = Info(name="小四", address="广东")
    db.session.add_all([people])
    db.session.commit()

def delete_data(id):
    try:
        people = Info.query.filter_by(id=id).first()
        db.session.delete(people)
        db.session.commit()
    except AttributeError:
        print("没有该人员信息,已删除或未建立")

def edit_data(id):
    try:
        people = Info.query.filter_by(id=id).first()
        people.name = "xiaoxiao"
        people.address = "广西"
        db.session.commit()
    except AttributeError:
        print("没有该人员信息")

def search_data(id):
    try:
        people = Info.query.filter_by(id=id).first()
        print("name:{}\naddress:{}\n".format(people.name, people.address))
    except AttributeError:
        print("没有该人员信息")


def query_all_data():
    try:
        infos = Info.query.all()
        print("infos: {}".format(infos))
        for info in infos:
            print("id: {}\nname: {}\naddress: {}\n".format(info.id, info.name, info.address))
            print("=====\n")
    except AttributeError:
        print("无人员信息")


def export_data_to_excel():
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet("人员信息", cell_overwrite_ok=True)
    field_num = 3
    field_datas = ["序号", "姓名", "地址"]
    for i in range(field_num):
        sheet.write(0, i, field_datas[i])
    try:
        infos = Info.query.all()
        for i in range(len(infos)):
            person_info = [infos[i].id, infos[i].name, infos[i].address]
            for j in range(field_num):
                sheet.write(i+1, j, person_info[j])
        workbook.save("people_infomation.xlsx")

    except AttributeError:
        print("没有人员信息")

def get_third_params():
    parser = ArgumentParser()
    parser.add_argument("--param")
    args = parser.parse_args()
    param = args.param
    return param


def queue_recursion(queue_data):
    '''recursion calculate 'AND': data[i]&data[i-1]&..&data[0]'''
    if queue_data.qsize() == 1:
        return queue_data.get()
    else:
        return queue_data.get()&queue_recursion(queue_data)

def import_excel_to_database():
    path_default = "people_infomation.xlsx"
    parser = ArgumentParser()
    parser.add_argument("--path", default=path_default)
    args = parser.parse_args()
    path = args.path
    workbook = xlrd.open_workbook(path)
    '''get all sheets'''
    sheets = workbook.sheets()
    '''people information sheets'''
    people_infos = sheets[0]
    row_nums = people_infos.nrows
    col_nums = people_infos.ncols
    validate_status = [False, False, False]
    end_status = False
    
    table_heads = people_infos.row_values(0)
    standard_table_heads = ["序号", "姓名", "地址"]
    for i in range(len(standard_table_heads)):
        if table_heads[i] == standard_table_heads[i]:
            validate_status[i] = True
        else:
            validate_status[i] = False
    '''validate import excel fromat same to standard'''
    queue_data = Queue(maxsize=10)
    for validate in validate_status:
        queue_data.put(validate)
    end_status = queue_recursion(queue_data)
    if end_status:
        for row in range(1, row_nums):
            people_infos_contents = people_infos.row_values(row)
            print("info contents: {}".format(people_infos_contents))
            add_people = Info(name=people_infos_contents[1], address=people_infos_contents[2])
            db.session.add_all([add_people])
            db.session.commit()
    else:
        print("数据表与模板不匹配,检查excel表格")





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
    # insert_data()
    '''delete one data by id'''
    # delete_data(6)
    '''edit one data by id'''
    # edit_data(5)
    '''search one data by id'''
    # search_data(6)
    '''query data in dababase for test'''
    query_all_data()
    '''export database data to excel'''
    export_data_to_excel()
    '''import excel data to database'''
    # import_excel_to_database()

