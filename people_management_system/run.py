from flask import Flask, request, Response, jsonify, Blueprint
from flask import render_template, redirect, url_for
from flask_paginate import Pagination, get_page_parameter
from datetime import datetime, timedelta


from flask_nav import Nav
from flask_nav.elements import *

from app import app, db, manager
from app.models.databases import People, IronTriangle, Project, CameraDevice
import requests, json
# from flask_apscheduler import APScheduler 
import schedule
from flask_cors import CORS
import config
import base64
from api.facebase_structure.endpoints.test_api import ns as test_namespace
from api.restplus import api, custom_ui
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

def log_save(log_info):
    with open("./logs/log_save.log", "a") as f:
        f.write(str(log_info)+'\n')

def bar_test() -> Bar:
    bar = Bar()
    bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
    bar.add_yaxis("商家B", [15, 25, 16, 55, 48, 8])
    bar.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"), 
                        legend_opts=opts.LegendOpts(is_show=True),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        # xaxis_opts=opts.AxisOpts(
                        #     type_="category",
                        #     is_scale=True,
                        #     grid_index=1
                        # ),
                        # yaxis_opts=opts.AxisOpts(
                        #     grid_index=1,
                        #     is_scale=True,

                        # )
                        )
    return bar

def line_test() -> Line:
    line = Line()
    line.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    line.add_yaxis("商家C", [5, 20, 36, 10, 75, 90])
    line.add_yaxis("商家D", [15, 25, 16, 55, 48, 8])
    line.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"), 
                        legend_opts=opts.LegendOpts(is_show=True, pos_left="20%"),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        datazoom_opts=[opts.DataZoomOpts(is_show=True,)],
                        # xaxis_opts=opts.AxisOpts(type_="category"),
                        # yaxis_opts=opts.AxisOpts(
                        #     is_scale=True
                        # )
                        )
    return line

def pie_test() -> Pie:
    pie = Pie()
    clothes = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
    data = [5, 20, 36, 10, 75, 90]
    pie.add(
        "",
        [list(pair_data) for pair_data in zip(clothes, data)],
        radius=["30%", "75%"],
        center=["25%", "50%"],
        rosetype="radius",
        label_opts=opts.LabelOpts(is_show=False),
    )
    pie.add(
        "",
        [list(pair_data) for pair_data in zip(clothes, data)],
        radius=["30%", "75%"],
        center=["75%", "50%"],
        rosetype="area"
    )
    return pie

def bar_test_2():
    bar2 = Bar()
    bar2.add_xaxis(["小一", "小二", "小三", "小四", "小五", "小六"])
    bar2.add_yaxis("商家C", [5, 10, 10, 10, 75, 90])
    bar2.add_yaxis("商家D", [15, 25, 16, 55, 48, 8])
    bar2.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题", pos_top="48%"), 
                        legend_opts=opts.LegendOpts(is_show=True, pos_top="48%"),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        datazoom_opts=opts.DataZoomOpts(is_show=False, pos_left="50%"))
    return bar2


def grid_test():
    line = Line()
    line.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    line.add_yaxis("商家C", [5, 20, 36, 10, 75, 90])
    line.add_yaxis("商家D", [15, 25, 16, 55, 48, 8])
    line.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"), 
                        legend_opts=opts.LegendOpts(is_show=True, pos_left="20%"),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        datazoom_opts=[opts.DataZoomOpts(is_show=True,
                            xaxis_index=[0, 1],
                            type_="slider",
                            range_start=0,
                            range_end=100
                        )],
                        xaxis_opts=opts.AxisOpts(type_="category"),
                        yaxis_opts=opts.AxisOpts(
                            is_scale=True
                        )
                        )
    # overlap_line = line.overlap(bar_test())
    bar = Bar()
    bar.add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
    bar.add_yaxis("商家A", [5, 20, 36, 10, 75, 90],xaxis_index=1, yaxis_index=1, label_opts=opts.LabelOpts(is_show=False))
    bar.add_yaxis("商家B", [15, 25, 16, 55, 48, 8])
    bar.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"), 
                        legend_opts=opts.LegendOpts(is_show=True),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        
                        xaxis_opts=opts.AxisOpts(
                            type_="category",
                            is_scale=True,
                            grid_index=1,
                            split_number=20,
                            axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                            axistick_opts=opts.AxisTickOpts(is_show=False),
                            splitline_opts=opts.SplitLineOpts(is_show=False),
                            axislabel_opts=opts.LabelOpts(is_show=False),
                            min_="dataMin",
                            max_="dataMax"
                        ),
                        yaxis_opts=opts.AxisOpts(
                            grid_index=1,
                            is_scale=True,
                            split_number=2,
                            axislabel_opts=opts.LabelOpts(is_show=False),
                            axisline_opts=opts.AxisLineOpts(is_show=False),
                            axistick_opts=opts.AxisTickOpts(is_show=False),
                            splitline_opts=opts.SplitLineOpts(is_show=False)
                        )
                        )
    grid = Grid()
    grid.add(line, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="50%"))
    grid.add(bar, grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="70%", height="16%"))

    return grid

def bar_test_3():
    data_a = [5, 10, 10, 10, 75, 90]
    data_b = [15, 25, 16, 55, 48, 8]
    bar = (
        Bar()
        .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
        .add_yaxis("商家A", data_a)
        .add_yaxis("商家B", data_b)
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"))
    )
    return bar

@people_show.route("/multiple_link", methods=["GET", "POST"])
def multiple_link():
    grid_test().render("./templates/multiple_test.html")
    return render_template("multiple_test.html")

@people_show.route("/analysis/dynamic", methods=["GET", "POST"])
def analysis_dynamic():
    data_c = [5, 10, 10, 10, 75, 90]
    data_d = [15, 25, 16, 55, 48, 8]
    bar2 = Bar()
    bar2.add_xaxis(["小一", "小二", "小三", "小四", "小五", "小六"])
    bar2.add_yaxis("商家C", data_c)
    bar2.add_yaxis("商家D", data_d)
    bar2.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"), 
                        legend_opts=opts.LegendOpts(is_show=True),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        datazoom_opts=opts.DataZoomOpts(is_show=True))

    grid = Grid()
    layer_bar1 = grid.add(bar_test(), grid_opts=opts.GridOpts(pos_bottom="60%"))
    layer_bar2 = grid.add(bar_test_2(), grid_opts=opts.GridOpts(pos_top="60%"))
    '''Draggable kit'''
    page = Page(layout=Page.DraggablePageLayout)
    # page = Page()
    # page.add(layer_bar1, layer_bar2)
    # page.add(bar, bar2)
    page.add(bar_test(), line_test(), pie_test())
    page.render("./templates/render_dynamic.html")
    # grid.add(line, grid_opts=opts.GridOpts(pos_right="55%"))
    # grid.render("./templates/grid_chart.html")
    # Page().add(grid).render()
    # bar.render("./templates/mybar_chart.html")
    # c.render()
    print("test pyecharts")
    # return render_template("mybar_chart.html")
    # return render_template("grid_chart.html")
    # Page().add(grid).render()
    # return Page().add(grid).render()
    # return Markup(c.render_embed())
    return render_template("render_dynamic.html")
@people_show.route("/analysis/static", methods=["GET", "POST"])
def analysis_static():
    data_c = [5, 10, 10, 10, 75, 90]
    data_d = [15, 25, 16, 55, 48, 8]
    bar2 = Bar()
    bar2.add_xaxis(["小一", "小二", "小三", "小四", "小五", "小六"])
    bar2.add_yaxis("商家C", data_c)
    bar2.add_yaxis("商家D", data_d)
    bar2.set_global_opts(title_opts=opts.TitleOpts(title="Bar实例", subtitle="子标题"), 
                        legend_opts=opts.LegendOpts(is_show=True),
                        toolbox_opts=opts.ToolboxOpts(is_show=True),
                        visualmap_opts=opts.VisualMapOpts(is_show=True),
                        tooltip_opts=opts.TooltipOpts(is_show=True),
                        datazoom_opts=opts.DataZoomOpts(is_show=True))

    grid = Grid()
    layer_bar1 = grid.add(bar_test(), grid_opts=opts.GridOpts(pos_bottom="60%"))
    layer_bar2 = grid.add(bar_test_2(), grid_opts=opts.GridOpts(pos_top="60%"))
    '''Draggable kit'''
    # page = Page(layout=Page.DraggablePageLayout)
    page = Page()
    # page.add(layer_bar1, layer_bar2)
    # page.add(bar, bar2)
    page.add(bar_test(), line_test(), pie_test())
    page.render("./templates/render_static.html")
    # grid.add(line, grid_opts=opts.GridOpts(pos_right="55%"))
    # grid.render("./templates/grid_chart.html")
    # Page().add(grid).render()
    # bar.render("./templates/mybar_chart.html")
    # c.render()
    print("test pyecharts")
    # return render_template("mybar_chart.html")
    # return render_template("grid_chart.html")
    # Page().add(grid).render()
    # return Page().add(grid).render()
    # return Markup(c.render_embed())
    return render_template("render_static.html")

@people_show.route("/")
def home():
    # return render_template("index.html", title_name="Welcome")
    return render_template("home.html", title_name="Welcome")
    # return render_template("index.html")
@people_show.route("/login")
def login():
    return render_template("login.html")

@people_show.route("/signup")
def signup():
    return render_template("signup.html")

limiter = Limiter(app, key_func=get_remote_address, default_limits=["2 per day", "10 per hour"])

@people_show.route("/people/show", methods=["GET", "POST"])
# @limiter.limit("1000 per day")
# @limiter.limit("100/hour")
# @limiter.limit("1/minute") 
def people_info_show():
        # bootstrap = Bootstrap(app)
    """Showing People information.
    We should use the function name: user_info_show in the other *.html file, 
    like pg.user_pagination(paginate, user.user_info_show) 
    """
    ip = request.remote_addr
    headers = request.headers
    agent = request.user_agent
    # real_ip = request.headers['HTTP_X-Real-Ip']
    beijing_time = datetime.now()
    utc_time = datetime.utcnow()
    user_number_per_page = 3
    people_count = People.query.count()
    page = request.args.get(get_page_parameter(), type=int, default=1)
    start_page = (page-1)*user_number_per_page
    end_page = start_page + user_number_per_page
    # query data
    people = People.query.paginate(page, per_page=10)
    print("request ip: {}, request time: {}".format(ip, str(beijing_time)))
    print("type of beijing time: {}".format(type(beijing_time)))
    print("Beijing time: {}, UTC time: {}".format(beijing_time, utc_time))
    print("request headers: {}".format(headers))
    print("real ip: {}".format(agent))
    api = url_for("people.people_info_show")
    log_info = {"ip":ip, "beijing_time":str(beijing_time), "utc_time":str(utc_time),
    "api":api}
    log_save(log_info)
    
    print("current api: {}".format(api))
    # return "success"
    # return render_template("users/show.html", users=users.items, pagination=users, totalNumber=user_count)
    return render_template("people/show.html", people=people.items, pagination=people, totalNumber=people_count, ip=ip)

@people_show.route("/user/add", methods=["GET", "POST"])
def people_create():
#     if request.method == "GET":
#         print("GET request")
#         return render_template("users/add.html")
#     elif request.method == "POST":
#         u_id = request.form.get("u_id")
#         u_name = request.form.get("u_name")
#         u_phone = request.form.get("u_phone")
#         '''single check data'''
#         # algorithm_companies = request.values.get("algorithm_company")
#         '''multiple check datas'''
#         algorithm_companies = request.values.getlist("algorithm_company")
#         print("algorithm company: {}".format(algorithm_companies))
#         print("u_id: {}".format(u_id))
#         user_info = User(u_id=u_id, u_name=u_name, u_phone=u_phone)
#         db.session.add(user_info)
#         db.session.commit()
#         users = User.query.all()
#         print("POST request")
#         # return render_template("users/show.html", users=users)
        return redirect(url_for("people.people_info_show"))

@people_show.route("/user/delete/<int:id>", methods=["GET", "POST"])
def people_delete(id):
#     user_info = User.query.filter_by(id=id).first()
#     db.session.delete(user_info)
#     db.session.commit()
#     # users = User.query.all()
    return redirect(url_for("people.people_info_show"))

@people_show.route("/user/edit/<int:id>", methods=["GET", "POST"])
def people_edit(id):
#     if request.method == "GET":
#         user_info = User.query.filter_by(id=id).first()
#         return render_template("users/edit.html", users=user_info)
#     elif request.method == "POST":
#         user_info = User.query.filter_by(id=id).first()
#         user_info.u_id = request.form.get("u_id")
#         user_info.u_name = request.form.get("u_name")
#         user_info.u_phone = request.form.get("u_phone")
#         db.session.commit()
        return redirect(url_for("people.people_info_show"))

@people_show.route("/user/search", methods=["GET", "POST"])
def people_search():
#     u_name = request.form.get("u_name")
#     if not u_name:
#         return redirect(url_for("user.user_info_show"))
#     else:
#         user_info  = User.query.filter_by(u_name=u_name).all()
#         user_number_per_page = 3
#         user_count = User.query.filter_by(u_name=u_name).count()
#         page = request.args.get(get_page_parameter(), type=int, default=1)
#         start_page = (page-1)*user_number_per_page
#         end_page = start_page + user_number_per_page

#         users = User.query.paginate(page, per_page=10)
#         # return "success"
#         # return render_template("users/show.html", users=users.items, pagination=users, totalNumber=user_count)
        return render_template("people/show.html", users=user_info, pagination=users, totalNumber=user_count)



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

    
    



def initializer_app(flask_app):
	configure_app(flask_app)
	faceapi = Blueprint('face', __name__, url_prefix='/api')
	api.init_app(faceapi)
	api.add_namespace(facesearch_namespace)
	flask_app.register_blueprint(faceapi)
	# manager.run()
	# flask_app.run(host='0.0.0.0', port=8090, debug=True)


# config_name = os.environ.get('') or 'default'
def nav_init(nav, app):
    nav.register_element('top', Navbar(
        View("人脸管理", "user.home"),
        View("主页", "user.home"),
        View("关于", "user.user_info_show"),
        Subgroup(
            "项目",
            View("项目1", "user.user_info_show"),
            Separator(),
            Text("我是一行字"),
            View("访客", "user.user_info_show")
        ),
        Link("用户", "http://localhost:8090/api/display/user/show"),
        View("About", "user.user_info_show")
    ))
    nav.init_app(app)
    # configuring the following fucniton to show.html
    # {{nav.top.render()}}

def create_app():
    app = Flask(__name__)
    
    nav = Nav()
    bootstrap = Bootstrap(app)
    configure_app(app)
	# blue = Blueprint('test',__name__,url_prefix='/blue')
    faceapi = Blueprint('face', __name__, url_prefix='/api')
    # limiter = Limiter(app, key_func=get_remote_address, default_limits=["2 per day", "10 per hour"])
    # limiter.limit("2/minute")(user_show)
    api.init_app(faceapi)
    nav_init(nav, app)
	# api.init_app(blue)
    api.add_namespace(test_namespace)
	# custom_ui()
    app.register_blueprint(faceapi)
    app.register_blueprint(people_show)
	# app.register_blueprint(blue)
    db.init_app(app)
    return app
app = create_app()


def add_test_data():
    people_1 = People(uu_id=1, people_code="post001", people_name="小红", people_origin="社招",
                        people_sex="男", people_work_age="2.5", people_position="数据分析师")
    people_2 = People(uu_id=2, people_code="post002", people_name="小红", people_origin="社招",
                        people_sex="男", people_work_age="2.5", people_position="数据分析师")
    people_3 = People(uu_id=3, people_code="post003", people_name="小红", people_origin="社招",
                        people_sex="男", people_work_age="2.5", people_position="数据分析师")

    iron_triangle_1 = IronTriangle(uu_id=1, iron_triangle_code="tri001", iron_triangle_name="产品经理")
    iron_triangle_2 = IronTriangle(uu_id=2, iron_triangle_code="tri002", iron_triangle_name="项目经理")
    iron_triangle_3 = IronTriangle(uu_id=3, iron_triangle_code="tri003", iron_triangle_name="研发经理")

    prj_1 = Project(uu_id=1, project_code="project001", project_name="项目1", project_describe="测试数据")
    prj_2 = Project(uu_id=2, project_code="project002", project_name="项目2", project_describe="测试数据")
    prj_3 = Project(uu_id=3, project_code="project003", project_name="项目3", project_describe="测试数据")
    
    camera_1 = CameraDevice(camera_device_code="camera001", camera_device_name="镜头1", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=1)

    camera_2 = CameraDevice(camera_device_code="camera002", camera_device_name="镜头1", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=1)
    camera_3 = CameraDevice(camera_device_code="camera003", camera_device_name="镜头2", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=2)
    camera_4 = CameraDevice(camera_device_code="camera001", camera_device_name="镜头2", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=2)
    
    camera_5 = CameraDevice(camera_device_code="camera002", camera_device_name="镜头3", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=3)
    camera_6 = CameraDevice(camera_device_code="camera003", camera_device_name="镜头3", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=3)
    camera_7 = CameraDevice(camera_device_code="camera003", camera_device_name="镜头3", camera_device_location="安徽省",
                                camera_device_describe="测试数据", project_id=3)

    people_1.ironTriangles = [iron_triangle_1, iron_triangle_2]
    people_2.ironTriangles = [iron_triangle_1, iron_triangle_3]
    people_3.ironTriangles = [iron_triangle_2]

    people_1.projects = [prj_1, prj_2, prj_3]
    people_2.projects = [prj_1, prj_2]
    people_3.projects = [prj_2, prj_3]                           
    

    db.session.add_all([people_1, people_2, people_2])
    db.session.add_all([iron_triangle_1, iron_triangle_2, iron_triangle_3])
    db.session.add_all([prj_1, prj_2, prj_3])
    db.session.add_all([camera_1, camera_2, camera_3, camera_4, camera_5, camera_6, camera_7])

    db.session.commit()

def query_test():
    people_info = People.query.filter(People.uu_id==1).first()
    print("people information: {}".format(people_info))
    print("type of data: {}".format(type(people_info)))
    print("name: {}".format(people_info.people_name))
    print("iron info: {}".format(people_info.ironTriangles))
    print("project info: {}".format(people_info.projects))
    
    for iron_info in people_info.ironTriangles:
        print("iron name: {}".format(iron_info.iron_triangle_name))
    for prj_info in people_info.projects:
        print("camera info: {}".format(prj_info.camera_devices.all()))
        print("project name: {}".format(prj_info.project_name))
        if prj_info.camera_devices.all():
            for camera in prj_info.camera_devices.all():
                print("camera name: {}".format(camera.camera_device_name))
        else:
            print("项目没有部署镜头")

    project_info = Project.query.filter(Project.uu_id==1).first()
    print("project info: {}".format(project_info))
    print("project name: {}".format(project_info.project_name))
    print("camera: {}".format(project_info.camera_devices.all()))
    for camera in project_info.camera_devices.all():
        print("camera name: {}".format(camera.camera_device_name))



if __name__ == "__main__":
	# scheduler = APScheduler()
	# print("++++++")
	# app.config.from_object(Config())
	# scheduler.init_app(app)
	# scheduler.start()
    # db.drop_all()
    db.create_all()
    '''migrate and upgrade'''
    # manager.run()
    '''clear data from database'''
    # clear_data()

    '''add data in database for test'''
    # add_test_data()

    
    '''query data in dababase for test'''
    # query_test()


	# initializer_app(app)
    app.run(host='0.0.0.0', port=8090, debug=True)
    
    # student = Student()
    # student_infos = Student.query.filter(Student.u_id=="001").first()
    # student_infos = Student.query.filter_by(u_id="001").first()
    # print("student informations: {}".format(student_infos.teachers[0].name))
    # print("student informations: {}".format(student_infos.teachers[1].name))
