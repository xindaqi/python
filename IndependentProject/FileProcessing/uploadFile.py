import requests, json, base64 
import os
from os.path import join as pjoin
from scipy import misc
import cv2, time
from progressbar import *

basedir = os.path.abspath(os.path.dirname(__name__))

url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add"
# access_token = "24.317f4a0917a10cacc449656ae771b87d.2592000.1543911481.282335-14652460"
access_token = "24.333255485187e3d9fa739dc0e70d10f4.2592000.1547876027.282335-14652460"

request_url = url + "?access_token=" + access_token

# def file_name(file_dir):
# 	fileNames = os.listdir(file_dir)
# 	fileNames.sort(key=lambda x:str(x[:-4]), reverse=False)
# 	a = len(fileNames)
# 	b = int(a/2)
# 	c = 0
# 	user_id = 0
# 	while True:
# 		if c <= b:
# 			for c in range(c+2):
# 				with open("image/"+fileNames[c], "rb") as f:
# 					image = base64.b64encode(f.read())
# 					str_image = str(image, encoding='utf8')
# 					data = {
# 					"image":str_image,
# 					"image_type":"BASE64",
# 					"group_id":"test",
# 					"user_id":user_id
# 					}
# 					print(fileNames[c])
# 					print(c)

# 					# res = requests.post(request_url, json=data)
# 					# res = res.json()
# 					# res = jsonify(res)
# 					# return res
# 			c += 1
# 			user_id += 1
# 			print("c:{}".format(c))
# 			print("user_id: {}".format(user_id))

# 		else:
# 			break

# file_name("image")



###########################
def uploadImage():
	fileNames = os.listdir("imageYF")
	fileNames.sort(key=lambda x:str(x[:-4]), reverse=False)
	a = len(fileNames)# 人脸个数
	b = int(3*(a/3-1))# 人脸组数
	c = 0# 人脸组数
	user_id = 0

	userNames = []
	for root, dris, files in os.walk("imageYF"):
		for file in files:

			userName, extensionName = os.path.splitext(file)
			userNames.append(userName)

	userNames.sort(reverse=False)
	# for userName in userNames:
	# 	userNames.append(userName[:-2])
	userInfos = []
	for userName in userNames:
		print(type(userName))
		print(userName[:-2])
		userInfos.append(userName[:-2])
	# print(userInfos)
	temp = list(set(userInfos))
	temp.sort(reverse=False)
	# print(temp)

	# print(userNames)

	while True:
		# n每组人脸数
		n = 3
		if c <= b:
			while n:
				with open("imageYF/"+fileNames[c], "rb") as f:
					image = base64.b64encode(f.read())
					str_image = str(image, encoding='utf8')
					data = {
					"image":str_image,
					"image_type":"BASE64",
					"group_id":"test",
					"user_id":temp[user_id]
					}

					res = requests.post(request_url, json=data)
					# print(c)
					print(fileNames[c])
					print(data['user_id'])
					# print(user_id)
					c += 1
				n -= 1


						# res = requests.post(request_url, json=data)
						# res = res.json()
						# res = jsonify(res)
						# return res
			user_id += 1
			print("c:{}".format(c))
			print("user_id: {}".format(user_id))
		else:
			break


#########################
img_dir = '/home/xdq/xinPrj/python/XHBDFaceRecognition/faceDatasheets/'  # 自己单独建的文件夹, 用于存放从lfw读取的图片
# data_dir是lfw数据集路径
def readImage(sourceDir, objectDir):
	count = 0
	for guy in os.listdir(sourceDir):
		person_dir = os.path.join(sourceDir, guy)
		for i in os.listdir(person_dir):
			# print("Image_name: {}".format(i))
			image_dir = os.path.join(person_dir, i)
			count += 1
			img = cv2.imread(image_dir)
			uploadImagePath = basedir + objectDir
			cv2.imwrite(uploadImagePath+i, img)
			# print("Path add name: {}".format(uploadImagePath+i))
			# print("Image full path: {}".format(image_dir))
	print("Image number: {}".format(count))

print("Absolute path: {}".format(basedir))

# imageDir = basedir + "/dataSheets/lfw/"
# objDir = "/tenThousands/"
# readImage(imageDir, objDir)

def readImageBatch(sourceDir, objectDir):
	count = 0
	for guy in os.listdir(sourceDir):
		if count > 1970:
			break
		else:
			person_dir = os.path.join(sourceDir, guy)
			print("Lenght of Images: {}".format(len(os.listdir(person_dir))))
			for i in os.listdir(person_dir):

				# print("Image_name: {}".format(i))
				image_dir = os.path.join(person_dir, i)
				count += 1
				img = cv2.imread(image_dir)
				uploadImagePath = basedir + objectDir
				cv2.imwrite(uploadImagePath+i, img)
				# print("Path add name: {}".format(uploadImagePath+i))
				# print("Image full path: {}".format(image_dir))
	print("Image number: {}".format(count))
	

# imageDir = basedir + "/dataSheets/lfw/"
# objDir = "/twoThousands/"
# readImageBatch(imageDir, objDir)


def loadImagetoBD(sourceDir, groupId):
	url = "https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add"
	access_token = "24.333255485187e3d9fa739dc0e70d10f4.2592000.1547876027.282335-14652460"
	request_url = url + "?access_token=" + access_token
	fileNames = os.listdir(sourceDir)
	fileNames.sort(key=lambda x:str(x[:-4]), reverse=False)
	# print("Files name : {}".format(fileNames))
	a = len(fileNames)# 人脸个数
	print("Face number: {}".format(a))
	
	user_id = 0

	userNames = []
	for root, dris, files in os.walk(sourceDir):
		for file in files:
			userName, extensionName = os.path.splitext(file)
			userNames.append(userName)

	userNames.sort(reverse=False)
	# # for userName in userNames:
	# # 	userNames.append(userName[:-2])
	userInfos = []
	for userName in userNames:
	# 	print(type(userName))
		# print(userName)
		userInfos.append(userName)
	# # print(userInfos)
	temp = list(set(userInfos))
	temp.sort(reverse=False)
	# print("User info: {}".format(temp))
	print("face Number: {}".format(len(temp)))
	
	widgets = ['上传完成:', Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]

	pbar = ProgressBar(widgets=widgets, maxval=10*len(temp)).start()

	for i in range(len(temp)):
		# print("face i :{}".format(i))
		# print("Temp : {}".format(temp[i]))
		with open(sourceDir+"/"+fileNames[i], "rb") as f:
			image = base64.b64encode(f.read())
			str_image = str(image, encoding='utf8')
			data = {
			"image":str_image,
			"image_type":"BASE64",
			"group_id":groupId,
			"user_id":temp[i]
			}
			# print("Image base64 code: {}".format(image))
			res = requests.post(request_url, json=data)

			print("上传第{}张图片.".format(i+1))
			pbar.update(i*10)
			time.sleep(5)
	pbar.finish()


# dirImage = "twoThousands"
dirImage = "tenThousands"
groupID = "tenThousandsImage"
loadImagetoBD(dirImage, groupID)





#########################



# def file_name(file_dir):
# 	for root, dirs, files in os.walk(file_dir):
# 		print(root)
# 		print(dirs)
# 		print(files)


# def file_name1(file_dir):
# 	fileName = []
# 	for root, dirs, files in os.walk(file_dir):
# 		for file in files:
# 			if os.path.splitext(file)[1] == '.png':
# 				fileName.append(os.path.join(root, file))
# 	# return fileName
# 	print(fileName)

# file_name("image")
# # file_name1("image")
# fileNames = []
# fileData = os.walk("image")
# for root, dirs, files in fileData:
# 	for file in files:
# 		if os.path.splitext(file)[1] == '.png':
# 			fileNames.append(os.path.join(root, file))

# print(fileNames)
# print("=================")
# a = 0
# user_id = 0


# for fileName in fileNames:
# 	l = len(fileNames)
# 	# print(l)
# 	a += 1
# 	with open(fileName, "rb") as f:
# 		image = base64.b64encode(f.read())
# 		str_image = str(image, encoding='utf8')
# 		# print(str_image)
# 		for i in range(10):
# 			print("++",end='')
# 		if a % 2 == 0:
# 			user_id += 1
		



		# data = {
		# "image":str_image,
		# "image_type":"BASE64",
		# "group_id":"test",
		# "user_id":
		# }
		# res = requests.post(request_url, json=data)
	# print(fileName)
	# print(a)
	# print(user_id)


# def outputFileName(fileDir):
# 	a = 0 
# 	fileNames = os.listdir(fileDir)
# 	fileNames.sort(key=lambda x:str(x[:-4]), reverse=False)
# 	b = len(fileNames)
# 	c = int(b/2)
# 	i = 2*(c-1)
# 	# m = 2*m
# 	print(i)

# 	# fileNames[m:m+1]
# 	# for i in range(c):
# 	# 	fileNames[i:i+2]
# 	# 	print(i)

# outputFileName("image")



# def test():
# 	li = ['xsl', 'zll', 'xindaqi', 'zww', 'xinxiaoqi', 'xinerqi']
# 	a = len(li)# 6
# 	b = int(a/2)# 3
# 	c = 2*(b-1)# 4
# 	d = 0
# 	user_id = 0
# 	while True:

# 		if d <= c:
# 			user_id += 1 
# 			print(li[d:d+2])
# 			print(user_id)
# 			d = d + 2
# 		else:
# 			break
# 	# print(li[d:d+2])

# test()



# with open("image/Lo.png","rb") as f:
# 	image = base64.b64encode(f.read())
# 	str_image = str(image, encoding='utf8')
# 	print("=========================================")
# 	print(str(image, encoding='utf8'))
# 	data = {
# 	"image":str_image,
# 	"image_type":"BASE64",
# 	"group_id":"test",
# 	"user_id":"005"
# 	}
# 	res = requests.post(request_url, json=data)
# 	print(res)
# 	print(request_url)





# data = {
# 	"image":"",
#     "image_type":"BASE64",
# 	"group_id":"test",
# 	"user_id":"005"
# }

# res = requests.post(request_url, json=data)
# res = res.json()
# print(res)

