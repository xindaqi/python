from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS 

from app import app
from api.test_api.endpoints.data_trans import ns as test_namespace
from api.restplus import api 

CORS(app, supports_credentials=True)

def config_swagger(flask_app):
	flask_app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'
	flask_app.config['RESTPLUS_VALIDATE'] = True
	flask_app.config['RESTPLUS_MASK_SWAGGER'] =  False
	flask_app.config['SWAGGER_UI_OPERATION_ID'] = True
	flask_app.config['SWAGGER_UI_REQUEST_DURATION'] = True


	
# def create_app(flask_app):
# 	app = Flask(__name__)
# 	config_swagger(app)
# 	test_api_blue = Blueprint('test just', __name__, url_prefix='/api')
# 	api.init_app(test_api_blue)
# 	api.add_namespace(test_namespace)
# 	flask_app.register_blueprint(test_api_blue)
	# return app

def create_app(flask_app):
	config_swagger(flask_app)
	test_api_blue = Blueprint('test just', __name__, url_prefix='/api')
	api.init_app(test_api_blue)
	api.add_namespace(test_namespace)
	flask_app.register_blueprint(test_api_blue)

create_app(app)


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8088, debug=True)



