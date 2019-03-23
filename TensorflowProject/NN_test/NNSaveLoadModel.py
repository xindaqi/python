import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
# Ubuntu system font path
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

MODEL_SAVE_PATH = "./models"
MODEL_NAME_meta = "nn_model.ckpt"
MODEL_NAME_pb = "nn_model.pb"
LOG_DIR = "./logs/NNmergelog"
'''Simulation datas.'''
x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5*x_data + noise

'''Neural network structure.'''
input_size_1 = 1
output_size_1 = 10

input_size_2 = 10
output_size_2 = 1

def saved_model_meta():
	with tf.name_scope("Input"):
		'''Input layer: datas.'''
		xs = tf.placeholder(tf.float32, [None, 1], name='x')
		ys = tf.placeholder(tf.float32, [None, 1], name='y')

	with tf.name_scope("Layer1"):
		'''Hidden layer.'''
		weights_1 = tf.Variable(tf.random_normal([input_size_1, output_size_1]), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([1, output_size_1]), name='biases_1')
		layer_1 = tf.nn.relu(tf.matmul(xs, weights_1) + biases_1)
		# tf.summary.scalar(name, var)
		# tf.summary.scalar('layer_1', layer_1)
		tf.summary.histogram('weights_1', weights_1)
		tf.summary.histogram('biases_1', biases_1)
		tf.summary.histogram('layer_1', layer_1)

	with tf.name_scope("Output"):
		'''Output layer.'''
		weights_2 = tf.Variable(tf.random_normal([input_size_2, output_size_2]), name='weights_2')
		biases_2 = tf.Variable(tf.zeros([1, output_size_2]), name='biases_2')
		prediction = tf.matmul(layer_1, weights_2) + biases_2
		tf.summary.histogram('weights_2', weights_2)
		tf.summary.histogram('biases_2', biases_2)
		tf.summary.histogram('prediction', prediction)

	with tf.name_scope("Loss"):
		'''Loss function.'''
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
		tf.summary.scalar('loss', loss)
		tf.summary.histogram('loss', loss)

	with tf.name_scope("Train_Step"):
		'''Optimizer the model by loss.'''
		train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	'''
	Add the variable to the graph by collection.
	add_to_collection("name", value)
	'''
	tf.add_to_collection('predictions', prediction)
	tf.add_to_collection('loss', loss)
	merged = tf.summary.merge_all()

	'''Save model: instantiation.'''
	saver = tf.train.Saver()
	with tf.Session() as sess:
		'''Initialize the varaible and logs.'''
		summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		a = 0
		for i in range(300):
			'''Compute the nodes value.'''
			summary, train_step_value, loss_value, pre = sess.run([merged, train_step,loss,prediction], feed_dict={xs: x_data, ys: y_data})
			if i % 50 == 0:
				'''Get train parameters in every 50 steps.'''
				a += 1
				w1 = sess.run(weights_1)
				w2 = sess.run(weights_2)
				print("Weights_1 :{}".format(w1))
				print("weights_2 :{}".format(w2))
				# print(a)
				# loss_1 = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
				print("Loss :{}".format(loss_value))
				print(prediction)
				print(loss)
				print(train_step_value)
				pre = sess.run(prediction, feed_dict={xs: x_data})
			'''Save the model to specific files we before defined.'''
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME_meta))
			'''Write the total logs.'''
			summary_writer.add_summary(summary, i)
		summary_writer.close()


def saved_model_pb():
	'''Input layer.'''
	with tf.name_scope("Input"):
		xs = tf.placeholder(tf.float32, [None, 1], name='x')
		ys = tf.placeholder(tf.float32, [None, 1], name='y')
	'''Hidden layer.'''
	with tf.name_scope("Layer1"):
		weights_1 = tf.Variable(tf.random_normal([input_size_1, output_size_1]), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([1, output_size_1]), name='biases_1')
		layer_1 = tf.nn.relu(tf.matmul(xs, weights_1) + biases_1)
		tf.summary.histogram('weights_1', weights_1)
		tf.summary.histogram('biases_1', biases_1)
		tf.summary.histogram('layer_1', layer_1)

	'''Ouptput Layer.'''
	with tf.name_scope("Output"):
		weights_2 = tf.Variable(tf.random_normal([input_size_2, output_size_2]), name='weights_2')
		biases_2 = tf.Variable(tf.zeros([1, output_size_2]), name='biases_2')
		outputs_2 = tf.matmul(layer_1, weights_2)
		prediction = tf.add(outputs_2, biases_2, name="predictions")
		tf.summary.histogram('weights_2', weights_2)
		tf.summary.histogram('biases_2', biases_2)
		tf.summary.histogram('prediction', prediction)

	'''Loss function.'''
	with tf.name_scope("Loss"):
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
		tf.summary.scalar('loss', loss)
		tf.summary.histogram('loss', loss)

	'''Optimizer the loss.'''
	with tf.name_scope("Train_Step"):
		train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

	'''Merge all the summary up used.'''
	merged = tf.summary.merge_all()
	'''Save Model.'''
	with tf.Session() as sess:
		'''Initializer varabiles and log defined in Tensorflow.'''
		summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		a = 0
		for i in range(301):
			'''Convert nodes to constant in models by name.'''
			constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["Input/x", "Input/y", "Output/predictions"])
			'''Compute the nodes value and save the log file.'''
			summary, train_step_value, loss_value, pre = sess.run([merged, train_step, loss, prediction], feed_dict={xs: x_data, ys: y_data})
			if i % 50 == 0:
				'''Output train effects in every 50 steps.'''
				a += 1
				w1 = sess.run(weights_1)
				w2 = sess.run(weights_2)
				print("Weights_1 :{}".format(w1))
				print("weights_2 :{}".format(w2))
				# print(a)
				# loss_1 = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
				print("Loss :{}".format(loss_value))
				print(prediction)
				print(loss)
				print(train_step_value)
			'''Write the model parameters in specify files we are defined.'''
			with tf.gfile.FastGFile(os.path.join(MODEL_SAVE_PATH, MODEL_NAME_pb), mode="wb") as f:
				f.write(constant_graph.SerializeToString())
			'''Summary total logs in files.'''
			summary_writer.add_summary(summary, i)
		summary_writer.close()

def load_meta_model():
	x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis]
	noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
	y_data = np.square(x_data) - 0.5*x_data + noise

	saver = tf.train.import_meta_graph("./models/nn_model.ckpt.meta")
	model_params = tf.train.latest_checkpoint("./models/")

	with tf.Session() as sess:
		saver.restore(sess, model_params)
		'''
		Get defult graph structure, which operation must
		after loaded the model.
		'''
		g = tf.get_default_graph()
		'''Extract nodes in trained models by name.'''
		pre = g.get_collection("predictions")[0]
		x = g.get_tensor_by_name("Input/x:0")
		'''Load graph structure and parameters.'''
		pre = sess.run(pre, feed_dict={x: x_data})
		plt.figure(figsize=(6, 6))
		plt.plot(x_data, pre, label="预测结果")
		plt.grid()
		plt.xlabel("x轴", fontproperties=font)
		plt.ylabel("y轴", fontproperties=font)
		plt.scatter(x_data, y_data, s=10, c="r", marker="*", label="实际值")
		plt.legend(prop=font)
	'''Save and show image.'''
	plt.savefig("./images/meta_load.png", format="png")
	plt.show()

def load_pb_model():
	with tf.Session() as sess:
		'''Input data for evaluate the model.'''
		x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis]
		noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
		y_data = np.square(x_data) - 0.5*x_data + noise
		'''Load model from *.pb'''
		with gfile.FastGFile("./models/nn_model.pb", "rb") as f:
			new_graph = tf.GraphDef()
			new_graph.ParseFromString(f.read())
			tf.import_graph_def(new_graph, name='')
		'''
		Get default graph structure, which operation must be
		after loaded the modelself.
		'''
		g = tf.get_default_graph()
		'''
		Get tensor by name in graph we defined,
		we use the variable scope or name scope,
		thus we need append the name prefix before load node names.
		'''
		pre = g.get_tensor_by_name("Output/predictions:0")
		x = g.get_tensor_by_name("Input/x:0")
		'''Compute the prediction value by loading the trained model.'''
		pre = sess.run(pre, feed_dict={x: x_data})
		plt.figure(figsize=(6, 6))
		plt.plot(x_data, pre, label="预测结果")
		plt.grid()
		plt.xlabel("x轴", fontproperties=font)
		plt.ylabel("y轴", fontproperties=font)
		plt.scatter(x_data, y_data, s=10, c="r", marker="*", label="实际值")
		plt.legend(prop=font)
	'''Save and show image.'''
	plt.savefig("./images/pb_load.png", format="png")
	plt.show()

# saved_model_meta()
load_meta_model()
# saved_model_pb()
# load_pb_model()
# print("Test successful!")
