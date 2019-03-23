import requests, json, urllib
import os

ABS_DIR = os.path.abspath(os.path.dirname(__name__))
print("Absolute path: {}".format(ABS_DIR))

def get_data():
	i = 0
	classify = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	urls = []
	for i in range(10):
		for j in range(10):
			# print(i)
			url = "http://www.cs.toronto.edu/~kriz/cifar-10-sample/{}{}.png".format(classify[i], j+1)
			# response = requests.get(url)
			urls.append(url)

	print("data urls: {}".format(urls))
	return urls


def download_images(urls):
	for i, url in enumerate(urls):
		image_name = url.split('/')[-1]
		print("No.{} images is downloading".format(i))
		urllib.request.urlretrieve(url, "images/"+image_name)


if __name__ == "__main__":
	download_images(get_data())

