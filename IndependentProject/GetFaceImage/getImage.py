import requests, urllib, re 

def getData():
	i = 0
	URLs = []
	while(i < 60):
		i += 30
		print(i) 
		# baidu image url,queryWord is search keyword, which equal to word,
		# pn is page number, you can set a number to get the same number of image
		url = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%E4%BA%BA%E8%84%B8&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=0&ic=0&hd=0&latest=0&copyright=0&word=%E4%BA%BA%E8%84%B8&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&selected_tags=&pn={}".format(i)
		response = requests.get(url)
		response = response.text
		# # response = response.json()["data"]
		# print(type(response))

		# use regular expression to get image url, thuubURL is the real url to get the image
		# re.S you can get different line thumbURL
		picURL = re.findall('"thumbURL":"(.*?)",', response, re.S)
		# split joint url in a list
		URLs += picURL

		# print(type(str(picURL)))

		# URL.append(picURL)

		# print("Picture URL: {}".format(picURL))
	return URLs
	# for i, URL in enumerate(URLs):
	# 	print("num: {}, URL: {}".format(i, URL))
	# print("Length of picture URL: {}".format(len(URL)))
	# 	# print(type(picURL))
	# print("URL address: {}".format(URL))

	# 	# print("Image info : {}".format(response))

def downLoadImage(URLs):
	# get image number and the URL, like "0":"http://...."
	for i, URL in enumerate(URLs):
		print("No." + str(i + 1) + '.jpg' + 'is downloading...')
		# save images in the special directory
		urllib.request.urlretrieve(URL, "faceImages/face"+str(i+1)+'.jpg')
	# for i, URL in enumerate(URLs):
	# 	try:
	# 		pic = requests.get(URL, timeout=15)

	# 		string = "faceImages" + str(i + 1) + '.jpg'
	# 		with open(string, 'wb') as f:
	# 			f.write(pic.content)
	# 			print("Downloading {} image, URL:{}".format(str(i + 1), str(URL)))
	# 	except Exception as e:
	# 		print("Failed download {} image, URL: {}".format(str(i + 1), str(URL)))

if __name__ == "__main__":
	# getData()
	downLoadImage(getData())