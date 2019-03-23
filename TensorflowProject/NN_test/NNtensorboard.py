import tensorflow as tf 
import numpy as np 
import os
import matplotlib.pyplot as plt 


MODEL_SAVE_PATH = "./models"
MODEL_NAME = "model.ckpt"
LOG_DIR = "./logs"
# 模拟数据
x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis] 
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5*x_data + noise

# 神经网络结构
input_size_1 = 1
output_size_1 = 10

input_size_2 = 10
output_size_2 = 1


def _result_():
	# 输入层Input
	with tf.name_scope("Input"):
		xs = tf.placeholder(tf.float32, [None, 1], name='x')
		ys = tf.placeholder(tf.float32, [None, 1], name='y')

	# 神经网路第一层:Layer1
	with tf.name_scope("Layer1"):
		weights_1 = tf.Variable(tf.random_normal([input_size_1, output_size_1]), name='weights_1')
		biases_1 = tf.Variable(tf.zeros([1, output_size_1]), name='biases_1')
		layer_1 = tf.nn.relu(tf.matmul(xs, weights_1) + biases_1)	
	# 神经网络:输出层Output
	with tf.name_scope("Output"):
		weights_2 = tf.Variable(tf.random_normal([input_size_2, output_size_2]), name='weights_2')
		biases_2 = tf.Variable(tf.zeros([1, output_size_2]), name='biases_2')
		prediction = tf.matmul(layer_1, weights_2) + biases_2
	# 损失函数
	with tf.name_scope("Loss"):
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
	# 损失函数迭代优化
	with tf.name_scope("Train_Step"):
		train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	# 增加变量
	tf.add_to_collection('prediction', prediction)
	tf.add_to_collection('loss', loss)
	# 持久化模型
	saver = tf.train.Saver()
	# 训练
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		x = sess.run(xs, feed_dict={xs: x_data})
		y = sess.run(ys, feed_dict={ys: y_data})
		a = 0
		# 迭代
		for i in range(300):
			train_step_value, loss_value, pre = sess.run([train_step,loss,prediction], feed_dict={xs: x_data, ys: y_data})
			# 保存模型			
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=i)
		# 保存计算日志
		writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
		writer.close()

_result_()





	
		
			





