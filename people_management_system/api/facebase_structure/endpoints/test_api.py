import requests
from flask import request, json, jsonify, render_template, url_for, redirect
from flask_restplus import Resource
from api.facebase_structure.parsers import admin_test_parsers
from api.facebase_structure.parsers import user_info_parsers

from api.facebase_structure.jsons import admin_test_jsons
from api.facebase_structure.jsons import user_info_jsons

from flask_paginate import Pagination, get_page_parameter

# from api.facebase_structure.jsons import admin_test_jsons
# from api.facebase_structure.jsons import user_info_jsons
from flask_bootstrap import Bootstrap


from api.restplus import api
from flask_cors import CORS
from flask import Response
from app import app, db
from app.models.databases import People, IronTriangle, Project, CameraDevice

# bootstrap = Bootstrap(app)

ns = api.namespace('facebase', description="管理员接口^_^", ordered=False)
securities = {'security_two':['apikey', {'oauth2':'read'}], 'security_one':'apikey',
'security_zero':None}

# @ns.route('/admin', methods=["GET", "POST"])
# class AdminTest(Resource):
#     # @api.doc(security=['apikey', {'oauth2':'read'}])
#     # @api.doc(security='apikey')
#     # @api.vendor({
#     #     'extension-1':{'name':'admin'},
#     #     'x-extension-3':'x-prefix is optional'
#     # })

#     @api.expect(admin_test_parsers)
#     @api.doc(description="admin", security=securities['security_one'])
#     @api.doc(responses={200:'success', 250:'bazinga'})
#     def get(self):
#         """
#         获取数据
#         """
#         admin_name = request.args.get('admin_name')
#         admin_phone = request.args.get('admin_phone')
#         return jsonify({"admin_name":admin_name, "admin_phone":admin_phone})

# @ns.route("/user/add", methods=["GET", "POST"])
# class AddUser(Resource):
#     @api.expect(user_info_parsers)
#     @api.doc(description="user info", security=securities['security_one'])
#     @api.doc(response={200:'success'})
#     # def get(self):
#     def post(self):
#         """添加用户信息"""
#         user = User()
#         user.u_id = request.args.get('u_id')
#         user.u_name = request.args.get('u_name')
#         user.u_phone = request.args.get('u_phone')
#         db.session.add(user)
#         db.session.commit()
#         return jsonify({"code":"200", "description":"成功添加用户"})

# @ns.route("/user/delete", methods=["GET", "POST"])
# class DeleteUser(Resource):
#     @api.expect(user_info_parsers)
#     @api.doc(descriptoin="user info", security=securities["security_one"])
#     @api.doc(response={200:"success",201:"数据类型错误"})
#     def post(self):
#         """删除用户信息"""
#         u_id = request.args.get("u_id")
#         user_info = User.query.filter(User.u_id==u_id).first()
#         if user_info:
#             db.session.delete(user_info)
#             db.session.commit()
#             return jsonify({"code":200, "description":"删除成功"})
#         else:
#             return jsonify({"code":203, "description":"用户不存在或以删除"})

# @ns.route("/user/edit/<int:id>", methods=["GET", "POST"])
# class EditUser(Resource):
#     @api.expect(user_info_parsers)
#     @api.doc(description="user info", security=securities["security_one"])
#     @api.doc(response={200:"success", 203:"数据异常"})
#     def post(self, id):
#         """编辑用户信息"""
#         user_info = User.query.get(id)
#         u_id = request.args.get("u_id")
#         u_name = request.args.get("u_name")
#         u_phone = request.args.get("u_phone")
#         user_info.u_id = u_id
#         user_info.u_name = u_name
#         user_info.u_phone = u_phone
#         db.session.commit()
#         return jsonify({"code":200, "description":"编辑成功"})
        



# @ns.route("/user/search", methods=["GET", "POST"])
# class SerachUser(Resource):
#     # 4 get reqeust
#     # @api.expect(user_info_parsers)
#     # 4 post request
#     @api.expect(user_info_jsons)
#     @api.doc(description="user info", security=securities['security_one'])
#     @api.doc(response={200:'success'})
#     # def get(self):
#     def post(self):
#         """查询用户信息"""
#         # user = User()
#         # u_id = request.args.get('u_id')
#         # u_name = request.args.get('u_name')
#         # u_phone = request.args.get('u_phone')
#         # user_info = User.query.filter(User.u_id=="001").all()
#         # u_id = request.form["u_id"]
#         # u_name = request.args.get("u_name")

#         u_name = request.json.get("u_name")
#         user_info = User.query.filter_by(u_name=u_name).all()
#         if user_info:
#             user_count = len(user_info)
#             user_info_extract = [[user_by_one.u_id, user_by_one.u_name, user_by_one.u_phone] for user_by_one in user_info]
#             user_attribute = ["user_id", "user_name", "user_phone"]
            
#             '''this return format: users infomation merged by same attribute, like: {"user_id":[002, 003], "user_name":["xiaoxiao", "xiaoxiao"]}'''
#             # return jsonify({"code":"200", "user_count": user_count, "user_info":{"user_id":[user_id.u_id for user_id in user_info],
#             #         "user_name":[user_name.u_name for user_name in user_info], "user_phone":[user_phone.u_phone for user_phone in user_info]}})

#             '''this return format: users infomation one by one, like: a:{"id":"001", "name":"xiaoxiao"}'''
#             return jsonify({"code":"200", "user_count": user_count,
#             "user_info":[dict(zip(user_attribute, user_info_extract_one)) for user_info_extract_one in user_info_extract]})
#         else:
#             return jsonify({"code":203,"description":"用户不存在"})




# @ns.route("/user/login", methods=["GET", "POST"])
# # def login():
# #     return render_template("login.html")
# class TestShow(Resource):
#     def get(self):
#         # return render_template("show.html")
#         login = render_template("login.html")
#         print("login data: {}".format(login))
#         print("----------------")
#         return login
#         # print("eval: {}".format(eval(login)))
#         # return eval(login)
#         # return "success connection"
    