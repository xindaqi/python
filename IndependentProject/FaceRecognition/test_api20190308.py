from PIL import Image, ImageDraw
import face_recognition as fr
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import base64

from app import app, db, manager
from app.models.databases import FaceString, FaceFloat, FaceText 


# app = Flask(__name__)

CORS(app, supports_credentials=True)

'''Load image and return image code'''
@app.route("/load_image", methods=["POST"])
def load_image():
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/test_image.jpg")
		# with open("./upload_images/test_image.jpg", 'rb') as f:
		image = fr.load_image_file("./upload_images/test_image.jpg")
		'''[[[1,3, 3],[23,34, 3]]] shape(1, 2, 3)'''
		print("image code: {}".format(image))
		'''image shape: (240, 427, 3),height=240, width=427, channels=3'''
		print("image shape: {}".format(image.shape))
		return jsonify({"data":{"image_code":image.tolist(), "shape":image.shape}})
	else:
		return jsonify({"error_code":250, "error_msg":"image format is wrong"})

'''Face landmarsk:get key points'''
@app.route("/face_landmarks", methods=['POST'])
def face_landmarks():
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/test_image.jpg")
		image = fr.load_image_file("./upload_images/test_image.jpg")
		face_landmarks_list = fr.face_landmarks(image)
		return jsonify({"data":{"face_landmarks":face_landmarks_list}})
	else:
		return jsonify({"error_code":250, "error_msg":"image format is wrong"})

'''Face encoding: get face feature which was already calculated.'''
@app.route("/face_encodings", methods=['POST'])
def face_encodings():
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/test_image.jpg")
		image = fr.load_image_file("./upload_images/test_image.jpg")

		face_encodings = fr.face_encodings(image)
		if face_encodings:
			face_encoding = fr.face_encodings(image)[0]
			return jsonify({"data":{"face_encodings":face_encodings.tolist(), "shape":face_encoding.shape}})
		else:
			return jsonify({"error_code":250, "error_msg":"图片中未找到人脸,请检查图片"})
	else:
		return jsonify({"error_code":250, "error_msg":"图片格式错误"})

'''Get face locations'''
@app.route("/face_locations", methods=["POST"])
def face_loacations():
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/face_locations.jpg")
		image_load = fr.load_image_file("./upload_images/face_locations.jpg")
		face_locations = fr.face_locations(image_load, number_of_times_to_upsample=1, model="cnn")
		'''[(140, 318, 310, 148)]'''
		print(face_locations)
		return "success"
'''Get face feature in database stored'''
@app.route("/get_face_feature", methods=["POST"])
def get_face_feature():
	if request.files and "image" in request.files:
		image = request.files["image"]
		image.save("./upload_images/new_image.jpg")
		image = fr.load_image_file("./upload_images/new_image.jpg")
		face_features = fr.face_encodings(image)
		if face_features:
			face_feature = face_features[0]
			infos = FaceText.query.all()
			saved_face_feature = [face.face_feature for face in infos]
			return jsonify({"data":{"face_feature":saved_face_feature}})
		else:
			return jsonify({{"error_code":250, "error_msg":"图片中未找到人脸,请检查图片"}})
	elif request.json and "image" in request.json:
		image = reqeust.json["image"]
		with open("./upload_images/new_image.jpg", "wb") as fdecode:
			fdecode_image = base64.b64decode(image)
			fdecode.write(fdecode_image)
		image = fr.load_image_file("./upload_images/new_image.jpg")
		face_features = fr.face_encodings(image)
		if face_features:
			face_feature = face_features[0]
			infos = FaceText.query.all()
			saved_face_feature = [face.face_feature for face in infos]
			return jsonify({"data":{"face_feature":saved_face_feature}})
		else:
			return jsonify({{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})
	else:
		return jsonify({"error_code":250, "error_msg":"图片格式错误"})


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
		'''judge has face or not in image uploaded'''
		if face_features:
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, number_of_times_to_upsample=0, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				face_texts = FaceText()
				face_texts.face_feature = str(face_feature.tolist())
				face_texts.name = name
				face_texts.sex = sex
				face_texts.address = address
				face_texts.project = project
				face_texts.position = position
				db.session.add(face_texts)
				db.session.commit()			
				return jsonify({"data":{"face location":{"top":top, "right":right, "bottom":bottom, "left":left}},"msg":"成功上传人脸"})
				# return "success"
			else:
				return jsonify({"error_code":250, "error_msg":"请上传只含一张人脸的图片"})
		else:
			return jsonify({"error_code":250, "error_msg":"图片中未检测到人脸信息,请确认上传图片中包含可辨识的人脸"})
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
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, number_of_times_to_upsample=0, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				face_texts = FaceText()
				face_texts.face_feature = str(face_feature.tolist())
				face_texts.name = name
				face_texts.sex = sex
				face_texts.address = address
				face_texts.project = project
				face_texts.position = position
				db.session.add(face_texts)
				db.session.commit()			
				return jsonify({"data":{"face location":{"top":top, "right":right, "bottom":bottom, "left":left}},"msg":"成功上传人脸"})
			else:
				return jsonify({"error_code":250, "error_msg":"请上传只含一张人脸的图片"})

		else:
			return jsonify({"error_code":250, "error_msg":"图片中未检测到人脸信息,请确认上传图片中包含可辨识的人脸"})
	else:
		return jsonify({"error_code":250, "error_msg":"图片格式错误"})


@app.route("/api/v1.0/single_face_compare", methods=["POST"])
def face_compare():
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
		'''get face feature'''
		face_features = fr.face_encodings(face_load)
		'''judge has face or not in uploaded image'''
		if face_features:
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, number_of_times_to_upsample=0, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				database_face = FaceText.query.all()
				database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
				database_name = [names.name for names in database_face]
				database_sex = [sexs.sex for sexs in database_face]
				# print("databae face feature: {}".format(database_face_feature))
				# print("type of database face feature: {}".format(type(database_face_feature)))
				'''temp dict save confidence '''
				temp_dict = {}
				# compare_result = fr.face_distance([image_1_feature], image_2_feature)
				compare_result = fr.face_distance(database_face_feature, face_feature)
				list_results = compare_result.tolist()
				print("list results: {}".format(list_results))
				for i in range(len(list_results)):
					temp_dict[i] = list_results[i]
				# print("list results: {}".format(temp_dict))
				'''from key to get min distance which is the most similiar one'''
				min_number = min(temp_dict, key=temp_dict.get)
				# print("minimum number: {}".format(min_number))
				'''get corresponing infos'''
				name_match = database_name[min_number]
				sex_match = database_sex[min_number]
				confidence_match = list_results[min_number]
				return jsonify({"data":{"compare_result":compare_result.tolist(), "user_info":{"name": name_match,"sex": sex_match,"confidence": 1-confidence_match},
						"face location":{"top":top, "right":right, "bottom":bottom, "left":left}}, "msg":"完成计算"})
				# return "success"
			else:
				return jsonify({"error_code":250, "error_msg":"检测到多个人脸,为确保安全,请保证镜头内单人验证"})
		else:
			return jsonify({{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})

	elif request.json and "image" in request.json:
		image = request.json["image"]
		with open("./upload_images/compare_face.jpg", "wb") as fdecode:
			decode_image = base64.b64decode(image)
			fdecode.write(decode_image)
		face_load = fr.load_image_file("./upload_images/compare_face.jpg")
		'''get face feature'''
		face_features = fr.face_encodings(face_load)
		'''judge has face or not in uploaded image'''
		if face_features:
			if len(face_features) == 1:
				face_locations = fr.face_locations(face_load, number_of_times_to_upsample=0, model="cnn")
				top, right, bottom, left = face_locations[0]
				face_feature = face_features[0]
				database_face = FaceText.query.all()
				database_face_feature = [np.array(eval(face.face_feature)) for face in database_face]
				database_name = [names.name for names in database_face]
				database_sex = [sexs.sex for sexs in database_face]
				# print("databae face feature: {}".format(database_face_feature))
				# print("type of database face feature: {}".format(type(database_face_feature)))
				temp_dict = {}
				# compare_result = fr.face_distance([image_1_feature], image_2_feature)
				compare_result = fr.face_distance(database_face_feature, face_feature)
				list_results = compare_result.tolist()
				for i in range(len(list_results)):
					temp_dict[i] = list_results[i]
				print("list results: {}".format(temp_dict))
				min_number = min(temp_dict, key=temp_dict.get)
				print("minimum number: {}".format(min_number))
				name_match = database_name[min_number]
				sex_match = database_sex[min_number]
				confidence_match = list_results[i]
				return jsonify({"data":{"compare_result":compare_result.tolist(), "user_info":{"name": name_match,"sex": sex_match,"confidence": 1-confidence_match},
						"face location":{"top":top, "right":right, "bottom":bottom, "left":left}}, "msg":"完成计算"})
			else:
				return jsonify({"error_code":250, "error_msg":"检测到多个人脸,为确保安全,请保证镜头内单人验证"})
		else:
			return jsonify({{"error_code":250, "error_msg":"图片中未找到人脸,请确认上传图片中包含可辨识的人脸"}})

	else:
		return jsonify({"error_code":250, "error_msg":"图片格式错误"})


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8090, debug=True)
	# db.create_all()











