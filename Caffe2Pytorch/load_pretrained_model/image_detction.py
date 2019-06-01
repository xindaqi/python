from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.proto import caffe2_pb2
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import os
from caffe2.python import core, workspace, models
import urllib.request
import operator
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

CAFFE_MODELS = "./models"
IMAGE_LOCATION = "./images/flower.jpg"
MODEL = 'squeezenet', 'init_net.pb', 'predict_net.pb', 'ilsvrc_2012_mean.npy', 227
codes =  "https://gist.githubusercontent.com/aaronmarkham/cd3a6b6ac071eca6f7b4a6e40e6038aa/raw/9edb4038a37da6b5a44c3b5bc52e448ff09bfe5b/alexnet_codes"

CAFFE_MODELS = os.path.expanduser(CAFFE_MODELS)
MEAN_FILE = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[3])
if not os.path.exists(MEAN_FILE):
    print("No mean file found!")
    mean = 128
else:
    print("Mean file found!")
    mean = np.load(MEAN_FILE).mean(1).mean(1)
    mean = mean[:, np.newaxis, np.newaxis]
print("mean was set to: {}".format(mean))

INPUT_IMAGE_SIZE = MODEL[4]

INIT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[1])
PREDICT_NET = os.path.join(CAFFE_MODELS, MODEL[0], MODEL[2])
if not os.path.exists(INIT_NET):
    print("WARNING: " + INIT_NET + "not found!")
else:
    if not os.path.exists(PREDICT_NET):
        print("WARNING: " + PREDICT_NET + "not found!")
    else:
        print("All needed files found!")


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx: startx+cropx]
def rescale(img, input_height, input_width):
    aspect = img.shape[1]/float(img.shape[0])
    if(aspect>1):
        res = int(aspect*input_height)
        img_scaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        res = int(input_width/aspect)
        img_scaled = skimage.transform.resize(img, (res, input_height))
    if(aspect==1):
        img_scaled = skimage.transform.resize(img, (input_widht, input_height))
    return img_scaled
img = skimage.img_as_float(skimage.io.imread(IMAGE_LOCATION)).astype(np.float32)
print("Original Image Shape: {}".format(img.shape))
img = rescale(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image shape after rescaling: {}".format(img.shape))
plt.figure()
plt.imshow(img)
plt.title("缩放后的图片", fontproperties=font)
plt.savefig("./images/scaled.png", format='png')
plt.show()

img = crop_center(img, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)
print("Image shape after cropping: {}".format(img.shape))
plt.figure()
plt.imshow(img)
plt.title("中心剪裁后的图片", fontproperties=font)
plt.savefig("./images/crop_center.png", format="png")
plt.show()

# switch to CHW(HWC->CHW)
img = img.swapaxes(1, 2).swapaxes(0, 1)
print("CHW Image Shape: {}".format(img.shape))

plt.figure()
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(img[i])
    plt.axis("off")
    plt.title("RGB 通道 {}".format(i+1), fontproperties=font)
plt.savefig("./images/channels.png", format="png")
print(type(img))
# switch to BGR(RGB->BGR)
img = img[(2, 1, 0), :, :]

img = img * 225 - mean

img = img[np.newaxis, :, :, :].astype(np.float32)
print("NCHW image: {}".format(img.shape))

with open(INIT_NET, "rb") as f:
    init_net = f.read()
with open(PREDICT_NET, "rb") as f:
    predict_net = f.read()

p = workspace.Predictor(init_net, predict_net)
results = p.run({"data": img})
results = np.asarray(results)
print("results shape: {}".format(results.shape))
preds = np.squeeze(results)
curr_pred, curr_conf = max(enumerate(preds), key=operator.itemgetter(1))
print("Prediction: {}".format(curr_pred))
print("Confidence: {}".format(curr_conf))

import json
results = np.delete(results, 1)
index = 0
highest = 0
arr = np.empty((0, 2), dtype=object)
arr[:, 0] = int(10)
arr[:, 1:] = float(10)
for i, r in enumerate(results):
    i = i + 1
    arr = np.append(arr, np.array([[i, r]]), axis=0)
    if(r>highest):
        highest = r
        index = i
N = 5
topN = sorted(arr, key=lambda x: x[1], reverse=True)[:N]
print("Raw top {} results {}".format(N, topN))
topN_inds = [int(x[0]) for x in topN]
print("Top {} classes in order: {}".format(N, topN_inds))

json_data = "./datas/labels.json"

def save_label(codes, json_data):
    response = urllib.request.urlopen(codes)
    print("response: {}".format(response))
    response = response.read().decode('utf-8')
    response = eval(response)

    '''
    {
     0: 'tench, Tinca tinca',
     1: 'goldfish, Carassius auratus',
     2: 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias',
     3: 'tiger shark, Galeocerdo cuvieri',
     4: 'hammerhead, hammerhead shark',}
    '''
    # print("response: {}".format(response))
    print("type of response: {}".format(type(response)))
    
    # save label in file
    with open(json_data, 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False)
save_label(codes, json_data)
# read label from file
with open(json_data, "r") as f:
    labels = f.readlines()
# extract string from list
labels = labels[0]
# swicth str to dict
labels = eval(labels)
# print("labels: {}".format(type(data[0])))
# print("labels: {}".format(data))
# print("eval label: {}".format(eval(data[0])))
class_LUT = []
for k,v in labels.items():
    v = v.split(",")[0]
#     print("label: {}".format(v))
    class_LUT.append(v)

for n in topN:
    print("Model predicts '{}' with {} confidence".format(class_LUT[int(n[0])], float("{0:.2f}".format(n[1]*100))))

images = ["./images/cowboy-hat.jpg","./images/cell-tower.jpg", "./images/Ducreux.jpg",
         "./images/pretzel.jpg", "./images/orangutan.jpg", "./images/aircraft-carrier.jpg",
         "./images/cat.jpg"]
NCHW_batch = np.zeros((len(images), 3, 227, 227))
print("Batch Shape: {}".format(NCHW_batch.shape))
for i, curr_img in enumerate(images):
    img = skimage.img_as_float(skimage.io.imread(curr_img)).astype(np.float32)
    img = rescale(img, 227, 227)
    img = crop_center(img, 227, 227)
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = img[(2, 1, 0), :, :]
    img = img * 225 - mean
    NCHW_batch[i] = img
print("NCHW image: {}".format(NCHW_batch.shape))
results = p.run([NCHW_batch.astype(np.float32)])
results = np.asarray(results)
preds = np.squeeze(results)

print("Squeezed Predictions Shape, with batch size {}:{}".format(len(images), preds.shape))
for i, pred in enumerate(preds):
    print("Results for: {}".format(images[i]))
    curr_pred, curr_conf = max(enumerate(pred), key=operator.itemgetter(1))
    print("\t Prediction: {}".format(curr_pred))
    print("\t Class Name: {}".format(class_LUT[int(curr_pred)]))
    print("\t Confidence: {}".format(curr_conf))


