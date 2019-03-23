import requests 
from flask import request, json, jsonify
from flask_restplus import Resource
from flask_cors import CORS 

from api.restplus import api

from api.test_api.args_data import test_add_get_arg
# from api.test_api.get_data import test_get_arg 

# from api.test_api.parser import test_add_post_arg
from api.test_api.json_data import test_post_arg 



ns = api.namespace('user', description='user api')

@ns.route('/get_info', methods=['GET'])
class Testget(Resource):
	@api.expect(test_add_get_arg)

	def get(self):
		user_name = request.args.get('user_name')
		user_phone = request.args.get('user_phone')

		return jsonify({'user_name':user_name, 'user_phone':user_phone})

@ns.route('/post_info', methods=['POST'])
class Testpost(Resource):
	@api.expect(test_post_arg)

	def post(self):
		# user_addr = request.form['user_addr']
		# user_sex = request.form['user_sex']
		user_addr = request.json['user_addr']
		user_sex = request.json['user_sex']

		return jsonify({'user_addr':user_addr, 'user_sex':user_sex})





