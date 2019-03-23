from flask import Blueprint, jsonify, request,Response,Flask
import urllib.request
import urllib.parse
import requests
import json
from flask_restplus import Api,Resource,reqparse,fields

chatbot = Blueprint('chatbot',__name__, url_prefix='/api')

@chatbot.route('/chatbot/connect')
def connect():
    print('Connect success!')
    return 'connect success!'

# 1--获取机器人access_token
@chatbot.route('/chatbot/token',methods=['POST','GET'])
def chatbotToken():
    url_token = "https://open.sobot.com/open/platform/getAccessToken.json"
    sign = request.args.get('sign')
    createTime = request.args.get('createTime')
    expire = request.args.get('expire')
    appId = request.args.get('appId')
    data = {
    # 'appId':'6dbeeb8ee0f14a7c99dc9b81cfc69f8b',
    'appId':appId,
    'createTime':createTime,
    'sign':sign,
    'expire':expire
    # '8cf52ef2ec1b7fa4e440c205cc704bed'
    }
    res = requests.post(url_token, params=data)
    res = res.json().get("data")
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res

# 2--机器人单轮对话
@chatbot.route('/chatbot/dialog',methods=['POST','GET'])
def diaglog():
    url_question = "https://open.sobot.com/open/platform/api.json"
    # headerData = {"Content-type":"application/json"}
    access_token = request.args.get('access_token')
    question = request.args.get('question')
    robotFlag = request.args.get('robotFlag')
    sysNum = request.args.get('sysNum')
    partnerId = request.args.get('partnerId')
    data = {
    'action':'chat_user_robot_ask',
    'access_token':access_token,
    'data':{
    # 'sysNum':'17589cc5a99441f89df2b0d29fd585c0',
    'sysNum':sysNum,
    'question':question,
    'robotFlag':robotFlag,
    'partnerId':partnerId
    }
    }
    res = requests.post(url_question,json=data)
    # res = res.json().get('code')
    # print(type(res.json().get('data')))
    # just return answer
    # dict type
    res = res.json().get('data')
    # return res['answer']

    # return all results
    # json.dumps:dict to json but type is str
    res = json.dumps(res)
    res = Response(res, mimetype='application/json')

    return res

# 3--添加知识库对话
@chatbot.route('/chatbot/addDialog')
def addDialog():
    url_dialog = "https://open.sobot.com/open/platform/api.json"
    ################################
    access_token = request.args.get('access_token')
    robotFlag = request.args.get('robotFlag')
    questionTitle = request.args.get('questionTitle')
    matchFlag = request.args.get('matchFlag')
    answerDesc = request.args.get('answerDesc')
    questionTypeId = request.args.get('questionTypeId')
    usedFlag = request.args.get('usedFlag')
    auditStatus = request.args.get('auditStatus')
    ################################
    data = {
    'action':'add_robot_doc',
    'access_token':access_token,
    'data':{
    'robotFlag':robotFlag,
    'questionTitle':questionTitle,
    'questionTypeId':questionTypeId,
    'matchFlag':matchFlag,
    'answerDesc':answerDesc,
    "usedFlag":usedFlag,
    "auditStatus":auditStatus
    }
    }

    res = requests.post(url_dialog,json=data)
    # res = res.json().get(code)
    res = res.json().get('data')
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res



# 4--删除知识库对话
@chatbot.route('/chatbot/deleteDialog')
def deleteDialog():
    url_delete = "https://open.sobot.com/open/platform/api.json"
    access_token = request.args.get('access_token')
    docId = request.args.get('docId')
    robotFlag = request.args.get('robotFlag')
    data = {
    'action':'delete_robot_doc',
    'access_token':access_token,
    'data':{
    'docId':docId,
    'robotFlag':robotFlag
    }
    }

    res = requests.post(url_delete,json=data)
    res = res.json().get('data')
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res


# 5--修改知识库对话
@chatbot.route('/chatbot/editDialog')
def editDialog():
    url_edit = "https://open.sobot.com/open/platform/api.json"
    access_token = request.args.get('access_token')
    docId = request.args.get('docId')
    robotFlag = request.args.get('robotFlag')
    questionId = request.args.get('questionId')
    questionTitle = request.args.get('questionTitle')
    questionTypeId = request.args.get('questionTypeId')
    matchFlag = request.args.get('matchFlag')
    answerId = request.args.get('answerId')
    answerDesc = request.args.get('answerDesc')
    usedFlag = request.args.get('usedFlag')
    auditStatus = request.args.get('auditStatus')

    data = {
    'access_token':access_token,
    'action':'update_robot_doc',
    'data':{
    'docId':docId,
    'robotFlag':robotFlag,
    'questionId':questionId,
    'questionTitle':questionTitle,
    'questionTypeId':questionTypeId,
    'matchFlag':matchFlag,
    'answerId':answerId,
    'answerDesc':answerDesc,
    'usedFlag':usedFlag,
    'auditStatus':auditStatus,
    }
    }

    res = requests.post(url_edit, json=data)
    res = res.json().get('data')
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res

# 6--查询知识库对话
@chatbot.route('/chatbot/dialogQuery')
def dialogQuery():
    url_dialog = "https://open.sobot.com/open/platform/api.json"
    access_token = request.args.get('access_token')
    docId = request.args.get('docId')
    robotFlag = request.args.get('robotFlag')
    data = {
    'access_token':access_token,
    'action':'query_doc_detail',
    'data':{
    'docId':docId,
    'robotFlag':robotFlag
    }
    }

    res = requests.post(url_dialog, json=data)
    res = res.json().get('data')
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res


# 7--查询知识库分类
@chatbot.route('/chatbot/questionList')
def questionList():
    url_question = "https://open.sobot.com/open/platform/api.json"
    access_token = request.args.get('access_token')
    robotFlag = request.args.get('robotFlag')
    typeFlag = request.args.get('typeFlag')
    parentTypeId = request.args.get('parentTypeId')
    data = {
    'access_token':access_token,
    'action':'search_question_type_list',
    'data':{
    'parentTypeId':parentTypeId,
    'robotFlag':robotFlag,
    'typeFlag':typeFlag
    }
    }

    res = requests.post(url_question, json=data)
    res = res.json().get('data')
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res

# 8--查询知识库对话列表
@chatbot.route('/chatbot/dialogList')
def dialogList():
    url_dialog = "https://open.sobot.com/open/platform/api.json"
    access_token = request.args.get('access_token')
    keyFlag = request.args.get('keyFlag')
    questionTypeId = request.args.get('questionTypeId')
    robotFlag = request.args.get('robotFlag')
    pageNo = request.args.get('pageNo')
    data = {
    'access_token':access_token,
    'action':'search_doc_list',
    'data':{
    'keyFlag':keyFlag,
    'questionTypeId':questionTypeId,
    'robotFlag':robotFlag,
    'pageNo':pageNo
    }
    }

    res = requests.post(url_dialog, json=data)
    res = res.json().get('data')
    res = json.dumps(res)
    res = Response(res, mimetype="application/json")
    return res







