#-*-coding:utf-8-*-
import os
from os import path
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

'''Nerual Network structure.

:params INPUT_NODE: input data dimension
:params OUTPUT_NODE: output data dimension
:params LAYER1_NODE: hidden layer dimension
:params BATCH_SIZE: data number in one group
'''
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "./new_models"
MODEL_NAME = "model.ckpt"

if not os.path.exists(MODEL_SAVE_PATH):
	'''Make sure model save path exists, if not exists, create one.
	:params MODEL_SAVE__PATH: model save path
	'''
	os.makedirs(MODEL_SAVE_PATH)

def get_weight_variable(shape, regularizer):
	'''Initializing weights and biases.
	:params shape: data dimension
	:params regularizer: regularize the weight for model more robust

	returns:
	:params weights: initialized weights
	'''
	weights = tf.get_variable("weights", shape,
		initializer=tf.truncated_normal_initializer(stddev=0.1))
	if regularizer !=None:
		tf.add_to_collection('losses', regularizer(weights))
	return weights

def inference(input_tensor, regularizer):
	'''Network inference layer data calculate.
	:params input_tensor: input data for network
	:params regularizer: regularizer flag

	returns:
	layer2: Network output
	'''
	with tf.variable_scope('layer1'):
		weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
		biases = tf.get_variable("biases", [LAYER1_NODE],
			initializer=tf.constant_initializer(0.0))
		layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

	with tf.variable_scope('layer2'):
		weights = get_weight_variable(
			[LAYER1_NODE, OUTPUT_NODE], regularizer)
		biases = tf.get_variable(
			"biases", [OUTPUT_NODE],
			initializer=tf.constant_initializer(0.0))
		layer2 = tf.matmul(layer1, weights) + biases

	return layer2

def train(mnist, model_file, restore_model):
	'''Train Nerual Network with MNIST datasets.
	:params mnist: input datas extract from MNIST datasets
	:params model_file: model path for judge if trained model exists
	:params restore_model: flag, load model formmer saved or not
	'''

	'''Standard input and output placeholder.'''
	x = tf.placeholder(
		tf.float32, [None, INPUT_NODE], name='x_input')
	y_ = tf.placeholder(
		tf.float32, [None, OUTPUT_NODE], name='y_input')
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)

	'''Predicted result.'''
	y = inference(x, regularizer)
	tf.add_to_collection("y_pre", y)
	global_step = tf.Variable(0, trainable=False)


	'''Moviing average model.'''
	variable_averages = tf.train.ExponentialMovingAverage(
		MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(
		tf.trainable_variables())
	'''Loss function.'''
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits=y, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	'''Learning rate.'''
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples / BATCH_SIZE,
		LEARNING_RATE_DECAY)
	'''Optimizer.'''
	train_step = tf.train.GradientDescentOptimizer(learning_rate)\
	.minimize(loss, global_step=global_step)
	with tf.control_dependencies([train_step, variables_averages_op]):
		train_op = tf.no_op(name='train')

	'''Model save class instantiate.'''
	saver = tf.train.Saver()
	with tf.Session() as sess:
		'''Initializing train variable.'''
		tf.global_variables_initializer().run()
		'''Loading trained model if there is.'''
		if path.isfile(model_file+".meta") and restore_model:
			print("Reloading modle file before training.")
			saver.restore(sess, model_file)
		else:
			print("Without any model and strat training.")
		'''Training steps.'''
		for i in range(TRAING_STEPS):
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			_, loss_value, step = sess.run([train_op, loss, global_step],
				feed_dict={x: xs, y_: ys})

			'''Saving modle every 1000 steps.'''
			if i % 1000 == 0:
				#输出当前的训练情况:当前batch上的损失函数结果.
				print("After %d training step(s) ,loss on training"
					"batch is %g."%(step, loss_value))
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
def main(argv=None):
	mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
	model_file = path.join(MODEL_SAVE_PATH, MODEL_NAME)
	train(mnist, model_file, True)

def load_model():
	'''Loading trained model and predict result which data extract from test datasets.'''
	mnist = input_data.read_data_sets("./mnist_data", one_hot=True)
	images = mnist.test.images[0]
	images = tf.expand_dims(images, [0])	
	saver = tf.train.import_meta_graph("./new_models/model.ckpt.meta")
	model_params = tf.train.latest_checkpoint("./new_models")
	with tf.Session() as sess:
		images = sess.run(images)
		print("images shape: {}".format(images.shape))
		saver.restore(sess, model_params)
		g = tf.get_default_graph()
		pre = g.get_collection("y_pre")[0]
		x = g.get_tensor_by_name("x_input:0")
		'''Prediction value shape is (1, 10)'''
		pre = sess.run(pre, feed_dict={x: images})
		'''Get value coresponding number.'''
		pre_num = tf.argmax(pre, 1)
		pre_num = sess.run(pre_num)
		print("predicted value's shape: {}".format(pre.shape))
		print("predicted number: {}".format(pre_num[0]))

if __name__ == '__main__':
	'''This function for training model.'''
	# tf.app.run()
	'''This function for loading model and predict.'''
	load_model()
