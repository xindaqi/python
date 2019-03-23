import matplotlib.pyplot as plt 
import face_recognition as fr 
from PIL import Image, ImageDraw


# image = "./upload_images/face_locations.jpg"
image = "./upload_images/YaoMac.jpg"
# image = "./upload_images/all_star0.jpeg"
# image = "./upload_images/all_star1.jpeg"
# image = "./upload_images/all_star2.jpeg"
# image = "./upload_images/compare_face.jpg"

face_load = fr.load_image_file(image)
print("shape of face load: {}".format(face_load.shape))
# print("face load: {}".format(face_load))

face_locations = fr.face_locations(face_load, model="cnn")
# face_locations = fr.face_locations(face_load)

face_features = fr.face_encodings(face_load)
# print("face features: {}".format(face_features))
print("numbers of face features: {}".format(len(face_features)))

pil_image = Image.fromarray(face_load)
d = ImageDraw.Draw(pil_image, 'RGBA')
# top, right, bottom, left = face_locations[0]
# (118, 340, 341, 117)
# [(47, 333, 115, 265), (84, 191, 141, 134), (41, 412, 81, 373)]
print("face locations: {}".format(face_locations))


temp_box_dict = {}
a = [face_location for face_location in face_locations]
print("box value: {}".format(a))

# for i in range(len(face_locations)):
# 	temp_box[i] = [top, right, bottom, left for face_location in face_locations]
temp_box_list = []
for face_location in face_locations:
	top, right, bottom, left = face_location
	box_width = right - left
	box_height = bottom - top
	box_size = box_width + box_height
	temp_box_list.append(box_size)
	print("box width: {}, box height: {}".format(box_width, box_height))
	d.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], fill=(255, 0, 0), width=3)

for i in range(len(temp_box_list)):
	temp_box_dict[i] = temp_box_list[i]
max_box_number = max(temp_box_dict, key=temp_box_dict.get)
print("max box number: {}".format(max_box_number))
pil_image.show()





# face_image = face_load[top:bottom, left:right]

# pil_image = Image.fromarray(face_image)

# d.line([(117, 118), (117, 341), (340, 341), (340, 118), (117, 118)], fill=(255, 0, 0), width=3)









