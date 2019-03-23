from flask import Flask, jsonify, request
from flask_cors import CORS


from app import app, db, manager
from app.models.databases import Knowledge
import config



CORS(app, supports_credentials=True)

@app.route('/add_info', methods=["POST"])
def add_info():
	question = request.form['input']
	answer = request.form['output']
	
	question_in_table = Knowledge.query.filter(Knowledge.question==question).first()
	if question_in_table:
		return jsonify({"number":"250", "info":"question already exists in database, do not saved!"})
	else:
		knowledge = Knowledge()
		knowledge.question = question
		knowledge.answer = answer
		db.session.add(knowledge)
		db.session.commit()
		return jsonify({"info":"saved data","saved_data":{"question":question, "answer":answer}})

	data_in_table = Knowledge.query.get(1)
	# knowledge.question = question
	# knowledge.answer = answer
	# db.session.add(knowledge)
	# db.session.merge(knowledge)
	# db.session.commit()
	# return "add info success!"
	return jsonify({"question":question_in_table.question})
	print("type of data: {}".format(type(question_in_table)))
	print("data: {}".format(question_in_table.question))
	print("questin: {}, answer: {}".format(data_in_table.question, data_in_table.answer))
	# return question_in_table
	# return "success search"
@app.route('/show_data', methods=["GET"])
def show_data():
	datas_in_table = Knowledge.query.all()
	if datas_in_table:
		print("datas in table: {}".format(datas_in_table))
		print("type of data in tables: {}".format(type(datas_in_table)))
		questions = [question.question for question in datas_in_table]
		answers = [answer.answer for answer in datas_in_table]
		print("questions in tables: {}".format(questions))
		print("answers in tables: {}".format(answers))
		
		return jsonify({"data":{"questions":questions, "answers":answers}, "msg":"遍历成功"})
	else:
		return jsonify({"error_code":250, "error_msg":"数据库为空"})

@app.route("/delete_data/<int:id>", methods=["GET"])
def delete_data(id):
	datas_in_table = Knowledge.query.get(id)
	if datas_in_table:
		question = datas_in_table.question
		answer = datas_in_table.answer
		db.session.delete(datas_in_table)
		db.session.commit()
		# questions = [question.question for question in datas_in_table]
		# answers = [answer.answer for answer in datas_in_table]
		return jsonify({"data":{"questions":question, "answers":answer}, "msg":"删除成功"})
	else:
		return jsonify({"error_code":250, "error_msg":"已经删除"})

@app.route('/calculate_feature', methods=["POST"])
def calculate_feature():
	feature = request.form['feature']
	datas_in_table = Knowledge.query.all()
	# cal_result = [float(feature) - float(questions.question) for questions in datas_in_table]
	cal_result = [abs(float(feature) - float(questions.question)) for questions in datas_in_table]
	dict_temp = {}
	for i in range(len(cal_result)):
		dict_temp[i] = cal_result[i]
	min_num = min(dict_temp, key=dict_temp.get)
	min_value = dict_temp[min_num]
	question = [questions.question for questions in datas_in_table][min_num]
	answer = [answers.answer for answers in datas_in_table][min_num]
	# info = zip(cal_result, question, answer)
	# for i in info:
	# 	print("info data: {}".format(i[0]))

	# min_info = [a[0] for a in info]
	# print("min info: {}".format(min_info))
	# print(*info)
	# return jsonify({"data":{"confidence":cal_result}, "msg":"计算完成"})
	return jsonify({"data":{"min_value":min_value, "info":{"question":question, "answer":answer}}})
	# return "success"



if __name__ == "__main__":
	db.create_all()
	# manager.run()
	app.run(host='0.0.0.0', port=8092, debug=True)












