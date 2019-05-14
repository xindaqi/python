#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from genplate import *

#产生用于训练的数据
class OCRIter():
    def __init__(self,batch_size,height,width):
        super(OCRIter, self).__init__()
        self.genplate = GenPlate("./font/platech.ttf",'./font/platechar.ttf','./NoPlates')
        self.batch_size = batch_size
        self.height = height
        self.width = width


    def iter(self):
        data = []
        label = []
        for i in range(self.batch_size):
            num, img = gen_sample(self.genplate, self.width, self.height)
            data.append(img)
            label.append(num)
        data_all = data
        label_all = label
        return data_all,label_all   


def rand_range(lo,hi):
    return lo+r(hi-lo)

def gen_rand():
    name = ""
    label=[]
    label.append(rand_range(0,31))  #产生车牌开头32个省的标签
    label.append(rand_range(41,65)) #产生车牌第二个字母的标签
    for i in range(5):
        label.append(rand_range(31,65)) #产生车牌后续5个字母的标签

    name+=chars[label[0]]
    name+=chars[label[1]]
    for i in range(5):
        name+=chars[label[i+2]]
    return name,label

def gen_sample(genplate, width, height):
    num,label =gen_rand()
    img = genplate.generate(num)
    # print("img: {}".format(img))
    img = cv2.resize(img,(width,height))
    img = np.multiply(img,1/255.0) #[height,width,channel]
    #img = img.transpose(2,0,1)
    #img = img.transpose(1,0,2)
    return label,img        #返回的label为标签，img为深度为3的图像像素

if __name__ == "__main__":
    get_batch = OCRIter(8, 72, 272)
    data, label = get_batch.iter()
    print("data: {} label: {}".format(data, label))