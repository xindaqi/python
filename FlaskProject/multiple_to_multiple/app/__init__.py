from flask import Flask 
from flask_sqlalchemy import SQLAlchemy 
from flask_script import Manager, Server
from flask_migrate import Migrate, MigrateCommand
from flask_cors import CORS 

app = Flask(__name__)

CORS(app, supprots_credentials=True)
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@localhost:3306/school?charset=utf8"
app.config['SQLALCHMEY_TRACK_MODIFICATIONS'] = True 

db = SQLAlchemy(app)
migrate = Migrate(app, db)
manager = Manager(app)

manager.add_command('db', MigrateCommand)

from app.models import databases