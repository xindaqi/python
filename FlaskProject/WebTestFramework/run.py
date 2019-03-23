from flask import Blueprint
# from flask_login import LoginManager, UserMixin, login_user, login_required
from app import app
from app import db
from app.models.chatbot import chatbot
from app import manager
from api.chatbot.endpoints.gets import ns as chatbot_namespace
from api.restplus import api
import config
from flask_paginate import Pagination, get_page_parameter


def configure_app(flask_app):

	# flask_app.config['SERVER_NAME'] = config.FLASK_SERVER_NAME
	# flask_app.config['SQLALCHEMY_DATABASE_URI'] = config.SQLALCHEMY_DATABASE_URI
	print(config.SQLALCHEMY_DATABASE_URI)
	# flask_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
	flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = config.RESTPLUS_SWAGGER_UI_DOC_EXPANSION
	flask_app.config['RESTPLUS_VALIDATE'] = config.RESTPLUS_VALIDATE
	flask_app.config['RESTPLUS_MASK_SWAGGER'] = config.RESTPLUS_MASK_SWAGGER
	flask_app.config['ERROR_404_HELP'] = config.RESTPLUS_ERROR_404_HELP


def initializer_app(flask_app):
	configure_app(app)
	chatbotapi = Blueprint('chatbotapi',__name__,url_prefix='/api')
	api.init_app(chatbotapi)
	api.add_namespace(chatbot_namespace)
	flask_app.register_blueprint(chatbotapi)
	flask_app.register_blueprint(chatbot)
	
	# longin
	# login_manager = LoginManager()
	# login_manager.init_app(flask_app)



def running():
	initializer_app(app)
	app.run(host='0.0.0.0', port=8080, debug=True)
	# manager.run()

if __name__ == '__main__':
	db.create_all()
	running()