from PIL import Image, ImageDraw
import face_recognition as fr
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import base64
import heapq

from app import app, db, manager
from app.models.databases import FaceString, FaceFloat, FaceText 


CORS(app, supports_credentials=True)

@app.route("/api/v1.0/add_face_feature", methods=["POST"])
def add_face_feature():
	"""
		Add face to database API.
		You can upload image which format *.png, *jpg or base64 are supported.
	"""
	if request.files and "image" in request.files:
		image = request.files["image"]
		name = request.form["name"]
		sex = request.form["sex"]
		address = request.form["address"]
		project = request.form["project"]
		position = request.form["position"]
		image.save("./upload_images/add_face.jpg")
		face_load = fr.load_image_file("./upload_images/add_face.jpg")
		face_features = fr.face_encodings(face_load)
		'''Judge has face or not in image uploaded'''
		if face_features:
			'''Ensure single face in image'''
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
				face_texts = FaceText()
				face_texts.face_feature = str(face_feature.tolist())
				face_texts.name = name
				face_texts.sex = sex
				face_texts.address = address
				face_texts.project = project
				face_texts.position = position
				db.session.add(face_texts)
				db.session.commit()			
				return jsonify({"data":{"bbox":{"top":top, "right":right, "bottom":bottom, "left":left}},"msg":"成功上传人脸"})
			else:
				return jsonify({"data":{"error_code":250, "error_msg":"请上传只含一张人脸的图片"}})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未检测到人脸信息,请确认上传图片中包含可辨识的人脸"}})
	elif request.json and "image" in request.json:
		image = request.json["image"]
		name = request.json["name"]
		sex = request.json["sex"]
		address = request.json["address"]
		project = request.json["project"]
		position = request.json["position"]
		with open("./upload_images/add_face.jpg", "wb") as fdecode:
			decode_image = base64.b64decode(image)
			fdecode.write(decode_image)
		face_load = fr.load_image_file("./upload_images/add_face.jpg")
		face_features = fr.face_encodings(face_load)
		if face_features:
			'''Ensure single face in image'''
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
				face_texts = FaceText()
				face_texts.face_feature = str(face_feature.tolist())
				face_texts.name = name
				face_texts.sex = sex
				face_texts.address = address
				face_texts.project = project
				face_texts.position = position
				db.session.add(face_texts)
				db.session.commit()			
				return jsonify({"data":{"bbox":{"top":top, "right":right, "bottom":bottom, "left":left}},"msg":"成功上传人脸"})
			else:
				return jsonify({"data":{"error_code":250, "error_msg":"请上传只含一张人脸的图片"}})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未检测到人脸信息,请确认上传图片中包含可辨识的人脸"}})
	else:
		return jsonify({"data":{"error_code":250, "error_msg":"图片格式错误或参数拼写错误或未输入参数"}})


@app.route("/api/v1.0/single_face_compare", methods=["POST"])
def face_compare():
	"""
		Face compare and output confidence API,
		user can upload image which format *.png, *.jpg or base64 are supported. 
		Compare uploaded image with the database includes, 
		we can know whether the face in database or not, however, for higher safety
		compared image must contain single face or warning will be raised.
	"""
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/compare_face.jpg")
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		'''Get face feature'''
		face_features = fr.face_encodings(face_load)
		print("face features: {}".format(face_features))
		face_locations = fr.face_locations(face_load, model="cnn")
		print("face loctaions: {}".format(face_locations))
		'''Judge has face or not in uploaded image'''
		if face_features:
			'''Ensure single face in image.'''
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
				database_face = FaceText.query.all()
				database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
				database_name = [names.name for names in database_face]
				database_sex = [sexs.sex for sexs in database_face]
				'''
					Compare results are distance between database face and uploaded face, 
					the lower value corresponding the higher confidence, for describe the 
					resutls more intitive and easily understading we use the negtive value
					to show the result: 1 - results
				'''
				compare_results = fr.face_distance(database_face_feature, face_feature)
				'''Compare results which was two face similar value.'''
				list_results = compare_results.tolist()
				compare_results = compare_results.tolist()
				
				compare_results = [1-compare_result for compare_result in compare_results]
				'''Take the larger four value among processed compare resutls'''
				compare_results = heapq.nlargest(4, compare_results)
				'''
					Temp dict save confidence namely distance between two face,
					trans the list to dictionary, aimed at extract the most simlar against 
					number.
				'''
				temp_dict = {}
				for i in range(len(list_results)):
					temp_dict[i] = list_results[i]
				'''
					From key to get min distance which is the most similiar one, and 
					record the min value against number to get the result value and 
					query infomrmation in database.
				'''
				min_number = min(temp_dict, key=temp_dict.get)
				'''Get corresponing infos.'''
				name_match = database_name[min_number]
				sex_match = database_sex[min_number]
				confidence_match = list_results[min_number]
				return jsonify({"data":{"compare_results":compare_results, "user_info":{"name": name_match,"sex": sex_match,"confidence": 1-confidence_match},
						"bbox":{"top":top, "right":right, "bottom":bottom, "left":left}}, "msg":"完成计算"})
			else:
				return jsonify({"data":{"error_code":250, "error_msg":"检测到多个人脸,为确保安全,请保证镜头内单人验证"}})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})

	elif request.json and "image" in request.json:
		image = request.json["image"]
		with open("./upload_images/compare_face.jpg", "wb") as fdecode:
			decode_image = base64.b64decode(image)
			fdecode.write(decode_image)
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		'''Get face feature'''
		face_features = fr.face_encodings(face_load)
		'''Judge has face or not in uploaded image'''
		if face_features:
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
				database_face = FaceText.query.all()
				database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
				database_name = [names.name for names in database_face]
				database_sex = [sexs.sex for sexs in database_face]
				'''
					Compare results are distance between database face and uploaded face, 
					the lower value corresponding the higher confidence, for describe the 
					resutls more intitive and easily understading we use the negtive value
					to show the result: 1 - results
				'''
				compare_results = fr.face_distance(database_face_feature, face_feature)
				list_results = compare_results.tolist()
				compare_results = compare_results.tolist()
				compare_results = [1-compare_result for compare_result in compare_results]
				'''Take the larger four value among processed compare resutls'''
				compare_results = heapq.nlargest(4, compare_results)
				'''
					Temp dict save confidence namely distance between two face,
					trans the list to dictionary, aimed at extract the most simlar against 
					number.
				'''
				temp_dict = {}
				for i in range(len(list_results)):
					temp_dict[i] = list_results[i]
				print("list results: {}".format(temp_dict))
				'''
					From key to get min distance which is the most similiar one, and 
					record the min value against number to get the result value and 
					query infomrmation in database.
				'''
				min_number = min(temp_dict, key=temp_dict.get)
				print("minimum number: {}".format(min_number))
				'''Get corresponing infos.'''
				name_match = database_name[min_number]
				sex_match = database_sex[min_number]
				confidence_match = list_results[min_number]
				return jsonify({"data":{"compare_results":compare_results, "user_info":{"name": name_match, "confidence": 1-confidence_match},
						"bbox":{"top":top, "right":right, "bottom":bottom, "left":left}}, "msg":"完成计算"})
			else:
				return jsonify({"data":{"error_code":250, "error_msg":"检测到多个人脸,为确保安全,请保证镜头内单人验证"}})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})

	else:
		return jsonify({"data":{"error_code":250, "error_msg":"图片格式错误或参数拼写错误或未输入参数"}})


@app.route("/api/v1.0/multi_face_compare", methods=["POST", "GET"])
def multi_face_compare():
	"""
		Face compare and output confidence API,
		user can upload image which format *.png, *.jpg or base64 are supported. 
		Compare uploaded image with the database includes, 
		we can know whether the face in database or not.
	"""
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/compare_face.jpg")
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		face_features = fr.face_encodings(face_load)
		if face_features:
			face_locations = fr.face_locations(face_load)
			'''Store box size info'''
			temp_box_list = []
			for face_location in face_locations:
				top, right, bottom, left = face_location
				box_width = right - left
				box_height = bottom - top
				box_size = box_width + box_height
				temp_box_list.append(box_size)
			'''Store max size box number which corresponding face will be compare'''
			temp_box_dict = {}
			for i in range(len(temp_box_list)):
				temp_box_dict[i] = temp_box_list[i]
			max_box_number = max(temp_box_dict, key=temp_box_dict.get)
			face_feature = face_features[max_box_number]
			top, right, bottom, left = face_locations[max_box_number]
			'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
			database_face = FaceText.query.all()
			database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
			database_name = [names.name for names in database_face]
			database_sex = [sexs.sex for sexs in database_face]
			'''
				Compare results are distance between database face and uploaded face, 
				the lower value corresponding the higher confidence, for describe the 
				resutls more intitive and easily understading we use the negtive value
				to show the result: 1 - results
			'''
			compare_results = fr.face_distance(database_face_feature, face_feature)
			list_results = compare_results.tolist()
			compare_results = compare_results.tolist()
			compare_results = [1-compare_result for compare_result in compare_results]
			'''Take the larger four value among processed compare resutls'''
			compare_results = heapq.nlargest(4, compare_results)
			'''
				Temp dict save confidence namely distance between two face,
				trans the list to dictionary, aimed at extract the most simlar against 
				number.
			'''
			temp_dict = {}
			for i in range(len(list_results)):
				temp_dict[i] = list_results[i]
			print("list results: {}".format(temp_dict))
			'''
				From key to get min distance which is the most similiar one, and 
				record the min value against number to get the result value and 
				query infomrmation in database.
			'''
			min_number = min(temp_dict, key=temp_dict.get)
			print("minimum number: {}".format(min_number))
			'''Get corresponing infos.'''
			name_match = database_name[min_number]
			sex_match = database_sex[min_number]
			confidence_match = list_results[min_number]
			return jsonify({"data":{"compare_results":compare_results, "user_info":{"name": name_match, "confidence": 1-confidence_match},
						"bbox":{"top":top, "right":right, "bottom":bottom, "left":left}}, "msg":"完成计算"})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})
	elif request.json and "image" in request.json:
		image = request.json["image"]
		with open("./upload_images/compare_face.jpg", "wb") as fdecode:
			decode_image = base64.b64decode(image)
			fdecode.write(decode_image)
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		face_features = fr.face_encodings(face_load)
		if face_features:
			face_locations = fr.face_locations(face_load)
			'''Store box size info'''
			temp_box_list = []
			for face_location in face_locations:
				top, right, bottom, left = face_location
				box_width = right - left
				box_height = bottom - top
				box_size = box_width + box_height
				temp_box_list.append(box_size)
			'''Store max size box number which corresponding face will be compared with database face'''
			temp_box_dict = {}
			for i in range(len(temp_box_list)):
				temp_box_dict[i] = temp_box_list[i]
			max_box_number = max(temp_box_dict, key=temp_box_dict.get)
			face_feature = face_features[max_box_number]
			top, right, bottom, left = face_locations[max_box_number]
			'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
			database_face = FaceText.query.all()
			database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
			database_name = [names.name for names in database_face]
			database_sex = [sexs.sex for sexs in database_face]
			'''
				Compare results are distance between database face and uploaded face, 
				the lower value corresponding the higher confidence, for describe the 
				resutls more intitive and easily understading we use the negtive value
				to show the result: 1 - results
			'''
			compare_results = fr.face_distance(database_face_feature, face_feature)
			list_results = compare_results.tolist()
			compare_results = compare_results.tolist()
			compare_results = [1-compare_result for compare_result in compare_results]
			'''Take the larger four value among processed compare resutls'''
			compare_results = heapq.nlargest(4, compare_results)
			'''
				Temp dict save confidence namely distance between two face,
				trans the list to dictionary, aimed at extract the most simlar against 
				number.
			'''
			temp_dict = {}
			for i in range(len(list_results)):
				temp_dict[i] = list_results[i]
			print("list results: {}".format(temp_dict))
			min_number = min(temp_dict, key=temp_dict.get)
			print("minimum number: {}".format(min_number))
			'''Get corresponing infos.'''
			name_match = database_name[min_number]
			sex_match = database_sex[min_number]
			confidence_match = list_results[min_number]
			return jsonify({"data":{"compare_results":compare_results, "user_info":{"name": name_match, "confidence": 1-confidence_match},
						"bbox":{"top":top, "right":right, "bottom":bottom, "left":left}}, "msg":"完成计算"})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})
	else:
		return jsonify({"data":{"error_code":250, "error_msg":"图片格式错误或参数拼写错误或未输入参数"}})


@app.route("/api/v1.0/multi_face_info", methods=["POST", "GET"])
def multi_face_info():
	"""
		Face compare and output confidence API,
		user can upload image which format *.png, *.jpg or base64 are supported. 
		Compare uploaded image with the database includes, 
		we can know whether the face in database or not.
	"""
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/compare_face.jpg")
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		face_features = fr.face_encodings(face_load)
		if face_features:
			face_locations = fr.face_locations(face_load, model="cnn")
			'''Store box size info'''
			temp_box_lists = []
			for face_location in face_locations:
				temp_box_dicts = {}
				top, right, bottom, left = face_location
				temp_box_dicts["top"] = top
				temp_box_dicts["right"] = right
				temp_box_dicts["bottom"] = bottom
				temp_box_dicts["left"] = left
				temp_box_lists.append(temp_box_dicts)
			print("box lists: {}".format(temp_box_lists))	
			
			'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
			database_face = FaceText.query.all()
			database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
			database_name = [names.name for names in database_face]
			database_sex = [sexs.sex for sexs in database_face]
			'''
				Compare results are distance between database face and uploaded face, 
				the lower value corresponding the higher confidence, for describe the 
				resutls more intitive and easily understading we use the negtive value
				to show the result: 1 - results
			'''
			'''Multi face compare results.'''
			multi_results = []
			for face_number in range(len(face_features)):
				multi_dicts = {}
				compare_results = fr.face_distance(database_face_feature, face_features[face_number])
				list_results = compare_results.tolist()
				compare_results = compare_results.tolist()
				compare_results = [1-compare_result for compare_result in compare_results]
				'''Take the larger four value among processed compare resutls'''
				compare_results = heapq.nlargest(4, compare_results)
				'''
					Temp dict save confidence namely distance between two face,
					trans the list to dictionary, aimed at extract the most simlar against 
					number.
				'''
				temp_dict = {}
				for i in range(len(list_results)):
					temp_dict[i] = list_results[i]
				print("list results: {}".format(temp_dict))
				'''
					From key to get min distance which is the most similiar one, and 
					record the min value against number to get the result value and 
					query infomrmation in database.
				'''
				min_number = min(temp_dict, key=temp_dict.get)
				print("minimum number: {}".format(min_number))
				'''Get corresponing infos.'''
				name_match = database_name[min_number]
				sex_match = database_sex[min_number]
				'''Get face locations.'''
				face_box = temp_box_lists[face_number]
				confidence_match = list_results[min_number]
				multi_dicts["name"] = name_match
				multi_dicts["confidence"] = 1-confidence_match
				multi_dicts["bbox"] = face_box
				multi_dicts["compare_results"] = compare_results
				multi_results.append(multi_dicts)
			return jsonify({"data":{"results":multi_results}, "msg":"完成计算"})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})
	elif request.json and "image" in request.json:
		image = request.json["image"]
		with open("./upload_images/compare_face.jpg", "wb") as fdecode:
			decode_image = base64.b64decode(image)
			fdecode.write(decode_image)
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		face_features = fr.face_encodings(face_load)
		if face_features:
			face_locations = fr.face_locations(face_load, model="cnn")
			'''Store box size info'''
			temp_box_lists = []
			for face_location in face_locations:
				temp_box_dicts = {}
				top, right, bottom, left = face_location
				temp_box_dicts["top"] = top
				temp_box_dicts["right"] = right
				temp_box_dicts["bottom"] = bottom
				temp_box_dicts["left"] = left
				temp_box_lists.append(temp_box_dicts)
			print("box lists: {}".format(temp_box_lists))
			'''ORM: database against the class(FaceText()), which can operator the tables contains in databases.'''
			database_face = FaceText.query.all()
			database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
			database_name = [names.name for names in database_face]
			database_sex = [sexs.sex for sexs in database_face]
			'''
				Compare results are distance between database face and uploaded face, 
				the lower value corresponding the higher confidence, for describe the 
				resutls more intitive and easily understading we use the negtive value
				to show the result: 1 - results
			'''

			'''Multi face compare results.'''
			multi_results = []
			for face_number in range(len(face_features)):
				multi_dicts = {}
				compare_results = fr.face_distance(database_face_feature, face_features[face_number])
				list_results = compare_results.tolist()
				compare_results = compare_results.tolist()
				compare_results = [1-compare_result for compare_result in compare_results]
				'''Take the larger four value among processed compare resutls'''
				compare_results = heapq.nlargest(4, compare_results)
				'''
					Temp dict save confidence namely distance between two face,
					trans the list to dictionary, aimed at extract the most simlar against 
					number.
				'''
				temp_dict = {}
				for i in range(len(list_results)):
					temp_dict[i] = list_results[i]
				print("list results: {}".format(temp_dict))
				'''
					From key to get min distance which is the most similiar one, and 
					record the min value against number to get the result value and 
					query infomrmation in database.
				'''
				min_number = min(temp_dict, key=temp_dict.get)
				print("minimum number: {}".format(min_number))
				'''Get corresponing infos.'''
				name_match = database_name[min_number]
				sex_match = database_sex[min_number]
				'''Get face locations.'''
				face_box = temp_box_lists[face_number]
				confidence_match = list_results[min_number]
				multi_dicts["name"] = name_match
				multi_dicts["confidence"] = 1-confidence_match
				multi_dicts["bbox"] = face_box
				multi_dicts["compare_results"] = compare_results
				multi_results.append(multi_dicts)
			return jsonify({"data":{"results":multi_results}, "msg":"完成计算"})
		else:
			return jsonify({"data":{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})
	else:
		return jsonify({"data":{"error_code":250, "error_msg":"图片格式错误或参数拼写错误或未输入参数"}})


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8090, debug=True)
	# db.create_all()











