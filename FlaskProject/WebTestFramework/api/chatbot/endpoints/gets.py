from flask_restplus import Resource
from api.chatbot.parsers import pagination_arguments_token
from api.chatbot.parsers import pagination_arguments_dialog
from api.chatbot.parsers import pagination_arguments_addDialog
from api.chatbot.parsers import pagination_arguments_deleteDialog
from api.chatbot.parsers import pagination_arguments_editDialog
from api.chatbot.parsers import pagination_arguments_dialogQuery
from api.chatbot.parsers import pagination_arguments_questionList
from api.chatbot.parsers import pagination_arguments_dialogList
from api.chatbot.parsers import getTokenParams
from api.restplus import api
from flask import request, jsonify
import requests, json
ns = api.namespace('chatbot',description=' Operation about Chatbot get request! ^_^ ')


@ns.route('/gettoken', methods=['GET', 'POST'])
class GetToken(Resource):
	@api.expect(getTokenParams)
	def post(self):
		url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/group/getusers"
		access_token = request.json['access_token']
		request_url = url + "?access_token=" + access_token
		group_id = request.json['group_id']
		data = {
		"group_id":group_id
		}
		res = requests.post(request_url, json=data)
		res = res.json()
		res = jsonify(res)
		return res


@ns.route('/token')
class Token(Resource):
	@api.expect(pagination_arguments_token)
	def get(self):
		"""
		获取接入凭证access_token
		"""
		return 'success'


@ns.route('/dialog')
class Dialog(Resource):
	@api.expect(pagination_arguments_dialog)
	def get(self):
		"""
		单轮对话
		"""
		return 'success'
@ns.route('/addDialog')
class addDialog(Resource):
	@api.expect(pagination_arguments_addDialog)
	def get(self):
		"""
		添加知识库对话
		"""
		return 'success'

@ns.route('/deleteDialog')
class DeleteDialog(Resource):
	@api.expect(pagination_arguments_deleteDialog)
	def get(self):
		"""
		删除知识库对话
		"""
		return 'success'

@ns.route('/editDialog')
class EditDialog(Resource):
	@api.expect(pagination_arguments_editDialog)
	def get(self):
		"""
		修改知识库对话
		"""
		return 'success'


@ns.route('/dialogQuery')
class DialogQuery(Resource):
	@api.expect(pagination_arguments_dialogQuery)
	def get(self):
		"""
		查询知识库对话
		"""
		return 'success'

@ns.route('/questionList')
class QuestionList(Resource):
	@api.expect(pagination_arguments_questionList)
	def get(self):
		"""
		查询知识库分类
		"""
		return 'success'


@ns.route('/dialogList')
class DialogList(Resource):
	@api.expect(pagination_arguments_dialogList)
	def get(self):
		"""
		查询知识库对话列表
		"""
		return 'success'



