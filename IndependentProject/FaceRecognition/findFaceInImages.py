from PIL import Image
import face_recognition as FR 
import tensorflow as tf 
import matplotlib.pyplot as plt
import os

def imageProcess():

	image = FR.load_image_file("images/Mac.png")
	print(type(image))
	print(image.shape)
	# print(image)

def faceNum():
	image = FR.load_image_file("images/Mac.png")
	face_locations = FR.face_locations(image)
	print("Face Number: {}".format(len(face_locations)))
	print(type(face_locations))
	print(face_locations)




def result():
	image = FR.load_image_file("images/Mac.png")
	face_locations = FR.face_locations(image)
	print("Face Number: {}".format(len(face_locations)))
	for face_location in face_locations:
		top, right, bottom, left = face_location
		print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
		face_image = image[top:bottom, left:right]
		pil_image = Image.fromarray(face_image)
		pil_image.show()


def tensorImage():
	# get image name
	file_names = []
	extensionNames = []
	fileNames = os.listdir('images')
	print("Normal order: {}".format(fileNames))
	fileNames.sort(key=lambda x:str(x[:-4]), reverse=False)
	print("Forward order: {}".format(fileNames))
	for file_name in fileNames:
		fileName, extensionName = os.path.splitext(file_name)
		file_names.append(fileName)
		extensionNames.append(extensionName)
	print("Extension names: {}".format(extensionNames))
	print("File names: {}".format(file_names))


	with tf.Session() as sess:
		plt.figure(figsize=(16.0, 11.0))
		image_init = FR.load_image_file("images/" + file_names[5]+extensionNames[5])
		face_locations = FR.face_locations(image_init)
		if face_locations:

			# print("Image shape: {}".format(image.shape))
			image_height, image_width, dimensions = image_init.shape
			# print("Image height: {}, Image width: {}, Dimensions: {}".format(image_height, image_width, dimensions))
			# print("Face Locations Type: {}".format(type(face_locations)))

			# print("Face Locations: {}".format(face_locations))
			print("Face Number: {}".format(len(face_locations)))
			face_loc_list = []
			for face_location in face_locations:
				top, right, bottom, left = face_location
				# [(39, 225, 168, 96)]
				# print(face_locations)
				xmin, ymax, xmax, ymin = face_location
				loc = [xmin/image_height, ymin/image_width, xmax/image_height, ymax/image_width]
				face_loc_list.append(loc)
				# print(loc)
				
				# print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
				face_image = image_init[top:bottom, left:right]
				pil_image = Image.fromarray(face_image)
				# pil_image.show()
			# print("Face location group: {}".format(face_loc))

			# # (309, 389, 3)
			# # Face Location: Top: 39, Left: 96, Bottom: 168, Right: 225
			# # box_process = 
			# for i in range(len(face_loc_list)):

			batched = tf.expand_dims(tf.image.convert_image_dtype(image_init, tf.float32),0)
			# boxes = tf.constant([face_loc_list])
			boxes = tf.constant([face_loc_list])
			bounding_box = tf.image.draw_bounding_boxes(batched, boxes,name="Hello")
			plt.imshow(bounding_box[0].eval())
			# plt.ion()
			
			plt.title("Processed Result")
			plt.xlabel("y aix")
			plt.ylabel("x aix")
			# plt.tight_layout()
			# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.5, top=0.9)
			plt.subplots_adjust(hspace=0.1, wspace=0.1)
			plt.show()
			# plt.savefig('processed/'+file_names[0] + '.png', format='png')
		else:
			image_up_down = tf.image.flip_up_down(image_init).eval()
			face_locations = FR.face_locations(image_up_down)
			if face_locations:
				# print("Image shape: {}".format(image.shape))
				image_height, image_width, dimensions = image_up_down.shape
				# print("Image height: {}, Image width: {}, Dimensions: {}".format(image_height, image_width, dimensions))
				# print("Face Locations Type: {}".format(type(face_locations)))

				# print("Face Locations: {}".format(face_locations))
				print("Face Number: {}".format(len(face_locations)))
				# face_loc_list: every face location info
				face_loc_list = []
				for face_location in face_locations:
					top, right, bottom, left = face_location
					# [(39, 225, 168, 96)]
					# print(face_locations)
					xmin, ymax, xmax, ymin = face_location
					# loc: face location
					loc = [xmin/image_height, ymin/image_width, xmax/image_height, ymax/image_width]
					face_loc_list.append(loc)
					# print(loc)
					
					# print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
					face_image = image_up_down[top:bottom, left:right]
					pil_image = Image.fromarray(face_image)
					# pil_image.show()
				# print("Face location group: {}".format(face_loc))

				# # (309, 389, 3)
				# # Face Location: Top: 39, Left: 96, Bottom: 168, Right: 225
				# # box_process = 
				# for i in range(len(face_loc_list)):

				batched = tf.expand_dims(tf.image.convert_image_dtype(image_up_down, tf.float32),0)
				# boxes = tf.constant([face_loc_list])
				boxes = tf.constant([face_loc_list])
				bounding_box = tf.image.draw_bounding_boxes(batched, boxes,name="Hello")
				plt.imshow(bounding_box[0].eval())
				# plt.ion()
				
				plt.title("Processed Result")
				plt.xlabel("y aix")
				plt.ylabel("x aix")
				# plt.tight_layout()
				# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.5, top=0.9)
				plt.subplots_adjust(hspace=0.1, wspace=0.1)
				plt.show()
				# plt.savefig('processed/'+file_names[0] + '.png', format='png')
			else:
				image_transpose = tf.image.transpose_image(image_init).eval()
				face_locations = FR.face_locations(image_transpose)
				if face_locations:
					image_height, image_width, dimensions = image_transpose.shape
					print("Face Number: {}".format(len(face_locations)))
				# face_loc_list: every face location info
					face_loc_list = []
					for face_location in face_locations:
						top, right, bottom, left = face_location
						# [(39, 225, 168, 96)]
						# print(face_locations)
						xmin, ymax, xmax, ymin = face_location
						# loc: face location
						loc = [xmin/image_height, ymin/image_width, xmax/image_height, ymax/image_width]
						face_loc_list.append(loc)
						# print(loc)
						
						# print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
						face_image = image_transpose[top:bottom, left:right]
						pil_image = Image.fromarray(face_image)
						# pil_image.show()
					# print("Face location group: {}".format(face_loc))

					# # (309, 389, 3)
					# # Face Location: Top: 39, Left: 96, Bottom: 168, Right: 225
					# # box_process = 
					# for i in range(len(face_loc_list)):

					batched = tf.expand_dims(tf.image.convert_image_dtype(image_up_down, tf.float32),0)
					# boxes = tf.constant([face_loc_list])
					boxes = tf.constant([face_loc_list])
					bounding_box = tf.image.draw_bounding_boxes(batched, boxes,name="Hello")
					plt.imshow(bounding_box[0].eval())
					# plt.ion()
					
					plt.title("Processed Result")
					plt.xlabel("y aix")
					plt.ylabel("x aix")
					# plt.tight_layout()
					# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.5, top=0.9)
					plt.subplots_adjust(hspace=0.1, wspace=0.1)
					plt.show()
					# plt.savefig('processed/'+file_names[0] + '.png', format='png')
				else:
					image_up_down = tf.image.flip_up_down(image_transpose).eval()
					face_locations = FR.face_locations(image_up_down)
					if face_locations:
						image_height, image_width, dimensions = image_up_down.shape
						print("Face Number: {}".format(len(face_locations)))
					# face_loc_list: every face location info
						face_loc_list = []
						for face_location in face_locations:
							top, right, bottom, left = face_location
							# [(39, 225, 168, 96)]
							# print(face_locations)
							xmin, ymax, xmax, ymin = face_location
							# loc: face location
							loc = [xmin/image_height, ymin/image_width, xmax/image_height, ymax/image_width]
							face_loc_list.append(loc)
							# print(loc)
							
							# print("Face Location: Top: {}, Left: {}, Bottom: {}, Right: {}".format(top,left,bottom,right))
							face_image = image_up_down[top:bottom, left:right]
							pil_image = Image.fromarray(face_image)
							# pil_image.show()
						# print("Face location group: {}".format(face_loc))

						# # (309, 389, 3)
						# # Face Location: Top: 39, Left: 96, Bottom: 168, Right: 225
						# # box_process = 
						# for i in range(len(face_loc_list)):

						batched = tf.expand_dims(tf.image.convert_image_dtype(image_up_down, tf.float32),0)
						# boxes = tf.constant([face_loc_list])
						boxes = tf.constant([face_loc_list])
						bounding_box = tf.image.draw_bounding_boxes(batched, boxes,name="Hello")
						plt.imshow(bounding_box[0].eval())
						# plt.ion()
						
						plt.title("Processed Result")
						plt.xlabel("y aix")
						plt.ylabel("x aix")
						# plt.tight_layout()
						# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.5, top=0.9)
						plt.subplots_adjust(hspace=0.1, wspace=0.1)
						plt.show()
						# plt.savefig('processed/'+file_names[0] + '.png', format='png')
					else:
						print("Can't find Human Face!Please Take Again!")




def imageInfo():
	# for root, dirs, files in os.walk('images'):
	# 	print(files)
	file_names = []
	extensionNames = []
	fileNames = os.listdir('images')
	print("Normal order: {}".format(fileNames))
	fileNames.sort(key=lambda x:str(x[:-4]), reverse=False)
	print("Forward order: {}".format(fileNames))
	for file_name in fileNames:
		fileName, extensionName = os.path.splitext(file_name)
		file_names.append(fileName)
		extensionNames.append(extensionName)
	print("Extension names: {}".format(extensionNames[0][1:]))
	print("File names: {}".format(file_names))
	
def showImage():
	with tf.Session() as sess:
		image = FR.load_image_file("images/Leon.png")
		print("Image encode: {}".format(image))
		# image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		# print("Image encode from int to float: {}".format(image.eval()))
		flip_up_down = tf.image.flip_up_down(image)
		flip_left_right = tf.image.flip_left_right(flip_up_down.eval())
		flip_transpose = tf.image.transpose_image(flip_left_right.eval())
		print("Flip up and down: {}".format(flip_up_down.shape))
		print("Flip left and right: {}".format(flip_left_right.shape))
		print("Flip transpose: {}".format(flip_transpose.shape))
		plt.subplot(1,3,1).set_title("A")
		plt.imshow(flip_up_down.eval())
		plt.subplot(1,3,2)
		plt.imshow(flip_left_right.eval())
		plt.subplot(1,3,3)
		plt.imshow(flip_transpose.eval())

		plt.show()






		# face_locations = FR.face_locations(image.eval())
		# print("Face Numbers: {}".format(len(face_locations)))
		# print(type(image))
		# plt.imshow(image.eval())
		# plt.show()

		# face_locations = FR.face_locations(image)
		# print("Face Numbers: {}".format(len(face_locations)))
		# image = tf.image.flip_left_right()
		# image = tf.image.transpose_image()
		# plt.imshow(image)
		# plt.show()


def face_locations():
	image = FR.load_image_file('images/Leon.png')
	face_locations = FR.face_locations(image)
	if face_locations:
		print("a")
	else:
		print("Null")
	print(face_locations)

# imageProcess()
# faceNum()
# result()
tensorImage()
# imageInfo()
# showImage()
# face_locations()