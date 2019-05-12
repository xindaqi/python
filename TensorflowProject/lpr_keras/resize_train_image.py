import tensorflow as tf 
import os 
import time
from os.path import join
import matplotlib.pyplot as plt

def parse_function(filenames):
	img_bytes = tf.read_file(filenames)
	img_decode = tf.image.decode_jpeg(img_bytes, channels=3)
	return img_decode

images_rgb = []
def reshape_image(image_path, sess):
	'''Reshape image to specify size.

	:params image_path: source image path
	:params sess: tensorflow Session

	return:
	image_rgb: image rgb information
	'''
	imgs_name = os.listdir(image_path)
	png = imgs_name[0].lower().endswith("png")
	imgs_path = [join(image_path, f) for f in imgs_name]
	imgs_num = len(imgs_path)
	imgs_queue = tf.data.Dataset.from_tensor_slices(imgs_path)
	imgs_map = imgs_queue.map(parse_function)
	img_decode = imgs_map.make_one_shot_iterator().get_next()
	for i in range(imgs_num):
		img_type = img_decode.dtype
		if img_decode.dtype != tf.float32:
			img_decode = tf.image.convert_image_dtype(img_decode, dtype=tf.float32)
		img_decode = tf.image.resize_images(img_decode, [128, 128], method=0)
		if img_decode.dtype == tf.float32:
			img_decode = tf.image.convert_image_dtype(img_decode, dtype=tf.uint8)

		img_value = sess.run(img_decode)
		images_rgb.append(img_value)
		# img_shape = img_value.shape
	return images_rgb

if __name__ == "__main__":
	with tf.Session() as sess:
		start_time = time.time()
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		image_path = "./data/train"
		# image_path = "./data/images"
		# image_path = "./data/images_mini"
		imgs_rgb = reshape_image(image_path, sess)
		imgs_num = len(images_rgb)
		if not os.path.exists("./data/train_resized"):
			os.makedirs("./data/train_resized")
		print("images number: {}".format(imgs_num))
		for i in range(imgs_num):
			fig = plt.figure(figsize=(1.28, 1.28))
			# fig = plt.gcf()
			# plt.imshow(imgs_value[:,:,0], cmap="Greys_r")
			plt.imshow(imgs_rgb[i])
			plt.axis("off")
			plt.savefig("./data/train_resized/{}.png".format(i+1), format="png")
			# plt.show()
			# plt.close(fig)
			plt.close("all")
			print("{} image has been saved.".format(i+1))
		# try:
			# while not coord.should_stop():
			# 	pass
				
		# except tf.errors.OutOfRangeError:
		# 	print("Executive finished.")
		# finally:
		# 	coord.request_stop()
		# coord.join(threads)

		coord.request_stop()
		coord.join(threads)
		end_time = time.time()
		time_cost = end_time - start_time
		# times = str(time_cost)
		# times = times.split(".")


		# hours = time_cost // 3600
		# minutes = (time_cost / 3600 - hours) * 60
		# seconds = 
		print("Time costed: {}s".format(time_cost))
		