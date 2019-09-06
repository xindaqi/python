from flask import Flask, jsonify, request
from flask_cors import CORS

from app import app, db, manager
from app.models.databases import User
import config
# cross over
CORS(app, supports_credentials=True)

@app.route('/add_info', methods=["POST"])
def add_info():
    user = User()

	# 外部输入
    # 
    u_id = request.form['id']
    u_name = request.form['name']
    u_phone = request.form['phone']
	# 实例化数据库类
	
	# 更新表字段信息
    user.u_id = u_id
    user.u_name = u_name
    user.u_phone = u_phone
	# 添加信息
    db.session.add(user)
	# 提交信息
    db.session.commit()
    return "add info success!"


@app.route('/get_info', methods=["POST"])
def get_info():
	# 外部输入
    # 
    u_id = request.form['id']
    u_name = request.form['name']
    u_phone = request.form['phone']
	# 实例化数据库类
    # user_id = User.query.filter(User.u_id==u_id).first()
    # user_id = User.query.filter(User.u_id==u_id).all()
    '''filter_by'''

    user_info = User.query.filter_by(u_id=u_id).all()
    # print("user info: {}".format(user_id[0].u_id))
    # print("user info: {}".format(user_id[0].u_name))
    # print("type: {}".format(type(user_id)))
    return jsonify({"code":200, "user_info":{"user_id":user_info[0].u_id, "user_name":user_info[0].u_name, "user_phone":user_info[0].u_phone}})
    # return jsonify({"code":200, "user_info":user_id.u_id})

if __name__ == "__main__":
	db.create_all()
	# manager.run()
	app.run(host='0.0.0.0', port=8092, debug=True)
