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
def queue_threads():
	'''Start queue and multi-threads.
	if you cannot set stop edge it will keep runing. 
	'''
	try:
		while not coord.should_stop():
			for i in range(imgs_num):
				plt.figure(figsize=(1.28, 1.28))
				# plt.imshow(imgs_value[:,:,0], cmap="Greys_r")
				plt.imshow(imgs_rgb[i])
				plt.axis("off")
				plt.savefig("./output_images/{}.png".format(i+1), format="png")
				plt.show()
	except tf.errors.OutOfRangeError:
		print("Executive finished.")
	finally:
		coord.request_stop()
	coord.join(threads)

if __name__ == "__main__":
	with tf.Session() as sess:
		'''Open a coordinator'''
		coord = tf.train.Coordinator()
		'''Start queue and fill'''
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		start_time = time.time()
		image_path = "./images"
		imgs_rgb = reshape_image(image_path,None, sess)
		imgs_num = len(images_rgb)
		# print("images number: {}, image value: {}".format(imgs_num, imgs_rgb))
		for i in range(imgs_num):
			plt.figure(figsize=(1.28, 1.28))
			# plt.imshow(imgs_value[:,:,0], cmap="Greys_r")
			plt.imshow(imgs_rgb[i])
			plt.axis("off")
			plt.savefig("./output_images/{}.png".format(i+1), format="png")
			plt.show()
		end_time = time.time()
		time_costed = end_time - start_time
		print("time costed: {}".format(time_costed))