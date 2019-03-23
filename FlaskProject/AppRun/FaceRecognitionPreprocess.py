from PIL import Image
import face_recognition as FR 
import tensorflow as tf 
import matplotlib.pyplot as plt
import os
from flask import Flask,jsonify, request
import json, requests 
import base64, cv2

app = Flask(__name__)

@app.route('/connect', methods=['GET', 'POST'])
def conncet():
	return 'connect success'

@app.route('/uploadimage', methods=['GET', 'POST'])
def tensorImage():
	with tf.Session() as sess:
		# get base64 encode image
		img_b64 = request.json['image']
		# decode image to string, just for save
		img_decode = base64.b64decode(img_b64)
		# save image
		with open('./saved_image/test.png', 'wb') as f:
			f.write(img_decode)

		# opencv read image
		img_init = cv2.imread('./saved_image/test.png')
		# get face location coordinate
		face_locations = FR.face_locations(img_init)

		# judging whether image has face
		if face_locations:
			print("Face Number: {}".format(len(face_locations)))
			face_loc_list = []

			for face_location in face_locations:
				a, b, c, d = face_location
				Yellow = (0, 255, 255)

				cv2.rectangle(img_init, (d,a), (b,c), Yellow, 1)
			cv2.imshow('face recognition', img_init)
			cv2.waitKey(0)
		else:
			# up and down flip image
			image_up_down = tf.image.flip_up_down(img_init).eval()
			face_locations = FR.face_locations(image_up_down)
			if face_locations:
		
				print("Face Number: {}".format(len(face_locations)))
				face_loc_list = []

				for face_location in face_locations:
					a, b, c, d = face_location
					Yellow = (0, 255, 255)

					cv2.rectangle(image_up_down, (d,a), (b,c), Yellow, 1)
				cv2.imshow('face recognition', image_up_down)
				cv2.waitKey(0)
			else:
				# tranpose image
				image_transpose = tf.image.transpose_image(img_init).eval()
				face_locations = FR.face_locations(image_transpose)

				if face_locations:
		
					print("Face Number: {}".format(len(face_locations)))
					face_loc_list = []

					for face_location in face_locations:
						a, b, c, d = face_location
						Yellow = (0, 255, 255)

						cv2.rectangle(image_transpose, (d,a), (b,c), Yellow, 1)
					cv2.imshow('face recognition', image_transpose)
					cv2.waitKey(0)

				else:
					# up and down flip image
					image_up_down = tf.image.flip_up_down(image_transpose).eval()
					face_locations = FR.face_locations(image_up_down)

					if face_locations:
		
						print("Face Number: {}".format(len(face_locations)))
						face_loc_list = []

						for face_location in face_locations:
							a, b, c, d = face_location
							Yellow = (0, 255, 255)

							cv2.rectangle(image_up_down, (d,a), (b,c), Yellow, 1)
						cv2.imshow('face recognition', image_up_down)
						cv2.waitKey(0)
					else:
						print("Can't find Human Face!Please Take Again!")
		return str(face_locations)


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8888, debug=True)

