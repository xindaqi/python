import tensorflow as tf 
import numpy as np 
import os

import matplotlib.pyplot as plt 



MODEL_SAVE_PATH = "./models"
MODEL_NAME = "model.ckpt"
LOG_DIR = "./logs/NNlog"




input_size_1 = 1
output_size_1 = 10

input_size_2 = 10
output_size_2 = 1


def inference(xs):
	
	with tf.name_scope("Layer1"):
		weights_1 = tf.Variable(tf.random_normal([input_size_1, output_size_1]), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([1, output_size_1]), name='biases_1')
		layer_1 = tf.nn.relu(tf.matmul(xs, weights_1) + biases_1)	

	with tf.name_scope("Output"):
		weights_2 = tf.Variable(tf.random_normal([input_size_2, output_size_2]), name='weights_2')
		biases_2 = tf.Variable(tf.zeros([1, output_size_2]), name='biases_2')
		prediction = tf.matmul(layer_1, weights_2) + biases_2

	return prediction




def loadModel():
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('models/model.ckpt-299.meta')
		saver.restore(sess, tf.train.latest_checkpoint('models/'))
		print("Load Weights_1 from model: {}".format(sess.run('weights_1:0')))
		print("Load Weights_2 from model: {}".format(sess.run('weights_2:0')))
		print("Load Biases_1 from model: {}".format(sess.run('biases_1')))
		print("Load Biases_2 from model: {}".format(sess.run('biases_2')))



# loadModel()



	
		
			





