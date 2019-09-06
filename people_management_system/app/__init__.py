from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager, Server
from flask_migrate import Migrate, MigrateCommand
from flask_cors import CORS 
from flask_bootstrap import Bootstrap


app = Flask(__name__)
# bootstrap = Bootstrap(app)

CORS(app, supports_credentials=True)
# app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:helloworld@localhost:33066/faceDataxin?charset=utf8"
app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://root:123456@localhost:3306/peoplebase?charset=utf8"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)
migrate = Migrate(app, db)
manager = Manager(app)


manager.add_command('db', MigrateCommand)
# manager.add_command("runserver", Server(host="0.0.0.0", port=8080, use_debugger=True))

# bootstrap = Bootstrap(app)

from app.models import databases
