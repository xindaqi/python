from flask import Flask,url_for,request,redirect,render_template
from flask_sqlalchemy import SQLAlchemy 
from flask_script import Manager,Server
from flask_bootstrap import Bootstrap 
from flask_migrate import Migrate,MigrateCommand
import os
from flask_cors import CORS
from flask_login import LoginManager, login_required, login_user, logout_user

app = Flask(__name__)
CORS(app,supports_credentials=True)
# 数据库配置：连接数据库
# 数据库协议://用户名:密码@ip地址:端口号/数据库名
#mysql+pymysql://xindaqi:123@123.234.234/databasename
# app.config.from_object('config')
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:$Mynew510$@localhost:3306/webData?charset=utf8'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SECRET_KEY'] = '123456'
# 创建数据操作对象
db = SQLAlchemy(app)
# 数据库迁移
migrate = Migrate(app,db)
# 管理app
manager = Manager(app)

# 迁移命令行操作
manager.add_command('db',MigrateCommand)
# manager.add_command("runserver",Server(host="0.0.0.0",port=8080,use_debugger=True))
# bootstrap模型
bootstrap = Bootstrap(app)
#注册路由模块与数据库结构
from app.models import databases,views

# login_manager
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = u"请先登录"
login_manager.login_message_category = "info"






