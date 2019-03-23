#-*-coding:utf-8-*-
from flask import Flask, request, jsonify, render_template, make_response, json
import os
from flask_cors import CORS 
import tensorflow as tf 
from preprocessing import preprocessing_factory
import reader
import model
import time
import base64

tf.app.flags.DEFINE_string('w', '', 'kernel')
tf.app.flags.DEFINE_string('b', '', 'kernel')
tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')

FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)
CORS(app, supports_credentials=True)

basedir = os.path.abspath(os.path.dirname(__name__))

@app.route('/connect', methods=['GET', 'POST'])
def connect():
	return "Successful connection"

@app.route('/path', methods=['GET'])
def path():
	return "Abs path is" + basedir

# Limit upload image type
# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])

# def judgeImageType(filename):
# 	return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/uploadImage', methods=['POST'])
def uploadImage():
	if request.method == "POST":
		# # file = request.files['Image']
		upload_base64 = request.json['Image']
		# upload_base64 = upload_base64.replace('+','-').replace(r'/','_')
		
		with open("./img/xdqtest_resize.png", 'wb') as fdecode:
			decode_base64 = base64.b64decode(upload_base64)
			fdecode.write(decode_base64)

		modelname = request.json['modelname']
		size = []

		# resized image and save
		with tf.Session() as sess:

			image_raw_data = tf.gfile.FastGFile("./img/xdqtest_resize.png", 'rb').read()
			image_decode = tf.image.decode_png(image_raw_data)
			height_source, width_source, _ = sess.run(image_decode).shape
			size = [height_source, width_source]
			if height_source + width_source < 1000:
				image_type = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)
				image_resized = tf.image.resize_images(image_type, [int(size[0]*0.8), int(size[1]*0.8)], method=1)

				image_data = tf.image.convert_image_dtype(image_resized, dtype=tf.uint8)
				encode_image = tf.image.encode_png(image_data)
				if os.path.exists("process") is False:
					os.mkdir("process")
				with tf.gfile.GFile("./process/xdqtest_resize.png", 'wb') as fsave:
					fsave.write(sess.run(encode_image))

			elif height_source + width_source < 1600:
				image_type = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)
				image_resized = tf.image.resize_images(image_type, [int(size[0]*0.7), int(size[1]*0.7)], method=1)

				image_data = tf.image.convert_image_dtype(image_resized, dtype=tf.uint8)
				encode_image = tf.image.encode_png(image_data)
				if os.path.exists("process") is False:
					os.mkdir("process")
				with tf.gfile.GFile("./process/xdqtest_resize.png", 'wb') as fsave:
					fsave.write(sess.run(encode_image))
			else:
				image_type = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)
				image_resized = tf.image.resize_images(image_type, [int(size[0]*0.5), int(size[1]*0.5)], method=1)

				image_data = tf.image.convert_image_dtype(image_resized, dtype=tf.uint8)
				encode_image = tf.image.encode_png(image_data)
				if os.path.exists("process") is False:
					os.mkdir("process")
				with tf.gfile.GFile("./process/xdqtest_resize.png", 'wb') as fsave:
					fsave.write(sess.run(encode_image))

		# return jsonify({"image_base64":upload_base64})

		

		# base64_tensor = tf.convert_to_tensor(upload_base64, dtype=tf.string)
		# img_str = tf.decode_base64(base64_tensor)
		# img = tf.image.decode_image(img_str, channels=3)
		# with tf.Session() as sess:
		# 	img_value = sess.run([img])[0]
		# 	# return jsonify({"img_value":img_value})
		# 	# return img_value.shape
		# 	height = img_value.shape[0]
		# 	width = img_value.shape[1]



		
		# 	print("Image value: {}".format(img_value.shape[0]))
		# 	return "a"



		# file = request.files.get("Image")

		# modelname = request.form['model']
		# file.headers['Access-Control-Allow-Origin'] = '*'
		# modelname.headers['Access-Control-Allow-Origin'] = '*'
		path = basedir + "/img/"
		# file_path = path + file.filename
		# file_path = path + "test.png"
		# file.save(file_path)
		# return str(model)
		# return "Upload success"
		height = 0 
		width = 0
		# image_path = "img/"+file.filename
		
		image_path = "./process/xdqtest_resize.png"
		# return image_path
		with open(image_path, 'rb') as img:
			with tf.Session().as_default() as sess:
				if image_path.lower().endswith('png'):
					image = sess.run(tf.image.decode_png(img.read()))
				else:
					image = sess.run(tf.image.decode_jpeg(img.read()))
				height = image.shape[0]
				width = image.shape[1]
		tf.logging.info("Image size: {}, {}".format(width, height))
		with tf.Graph().as_default():
			with tf.Session().as_default() as sess:
				# return jsonify({"height":height, "width":width})     
	    # tf.logging.info('Image size:{},{}'.format(weight, height))

	    		# Read image data
				image_preprocessing_fn, _ = preprocessing_factory.get_preprocessing(
	    			FLAGS.loss_model,
	    			is_training=False)
				image = reader.get_image(image_path, height, width, image_preprocessing_fn)
					    		# Add batch dimension
				image = tf.expand_dims(image, 0)

				generated = model.net(image, training=False)
				generated = tf.cast(generated, tf.uint8)

				# Remove batch dimension
				generated = tf.squeeze(generated, [0])

				# Restore model variables
				saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
				sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
				# Use absolute path
				model_path = basedir + "/models/" + modelname
				saver.restore(sess, model_path)
				# return model_path
				# Make sure 'generated' directory exists
				# generated_file = 'generated/' + file.filename
				generated_file = 'generated/xdqtest.png'
				if os.path.exists('generated') is False:
					os.makedir('generated')


				# Generate and write image data to file
				with open("./" + generated_file, 'wb') as img:
					start_time = time.time()
					img.write(sess.run(tf.image.encode_jpeg(generated)))
					end_time = time.time()
					# return "Done"

				# recover image from source input
				recover_image("./" + generated_file, size[0], size[1], sess)


				# process_image = basedir + "/" + generated_file
				process_image = basedir + "/" + 'resized_image/resized.png'

				# Encode image to base64
				with open(process_image, 'rb') as img:
					image_base64 = img.read()
					image_base64 = base64.b64encode(image_base64)
					# response = make_response(image_base64)
					# response.headers['Access-Control-Allow-Origin'] = '*'
					# response.headers['Access-Control-Methods'] = 'POST'
					# response.headers['Access-Control-Allow-Headers'] = 'x-requested-with, content-type'
					# print(type(response))
					# jsonify({"Image_base64":image_base64})
					return jsonify({"Image_base64":image_base64})

					# return jsonify({"Image base64":image_base64})

		# 			# start_time = time.time()
		# 			# img.write(sess.run(tf.image.encode_jpeg(generated)))
		# 			# end_time = time.time()
		# 			# tf.logging.info("Elapsed time: {}s".format(end_time - start_time))
		# 			# tf.logging.info("Done. Please check {}.".format(generated_file))
		# 			# return "Elapsed time: {}s".format(end_time - start_time)


def recover_image(image_dir, height, width, sess):
	image_raw_data = tf.gfile.FastGFile(image_dir, 'rb').read()
	image_decode = tf.image.decode_png(image_raw_data)
	# height_source, width_source, _ = sess.run(image_decode).shape
	image_type = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)
	image_resized = tf.image.resize_images(image_type, [height, width], method=1)
	image_data = tf.image.convert_image_dtype(image_resized, dtype=tf.uint8)
	encode_image = tf.image.encode_png(image_data)
	if os.path.exists("resized_image") is False:
		os.mkdir("resized_image")

	with tf.gfile.GFile("./resized_image/resized.png", 'wb') as fresize:
		fresize.write(sess.run(encode_image))



# def recover_image(image_dir, height, width):
# 	with tf.Session() as sess:
# 		image_raw_data = tf.gfile.FastGFile(image_dir, 'rb').read()
# 		image_decode = tf.image.decode_png(image_raw_data)
# 		# height_source, width_source, _ = sess.run(image_decode).shape
# 		image_type = tf.image.convert_image_dtype(image_decode, dtype=tf.float32)
# 		image_resized = tf.image.resize_images(image_type, [height, width], method=0)
# 		image_data = tf.image.convert_image_dtype(image_resized, dtype=tf.uint8)
# 		encode_image = tf.image.encode_png(image_data)
# 		if os.path.exists("resized_image") is False:
# 			os.mkdir("resized_image")

# 		with tf.gfile.GFile("./resized_image/resized.png") as fresize:
# 			fresize.write(sess.run(encode_image))

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8096, debug=True)