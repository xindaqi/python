import tensorflow as tf 
import numpy as np 
import os

import matplotlib.pyplot as plt 
from pylab import mpl 
# 
mpl.rcParams["font.serif"] = ['simhei']

MODEL_SAVE_PATH = "./models"
MODEL_NAME = "model.ckpt"


x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis] 

noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)

y_data = np.square(x_data) - 0.5*x_data + noise


xs = tf.placeholder(tf.float32, [None, 1], name='x')
ys = tf.placeholder(tf.float32, [None, 1], name='y')

input_size_1 = 1
output_size_1 = 10

input_size_2 = 10
output_size_2 = 1


weights_1 = tf.Variable(tf.random_normal([input_size_1, output_size_1]), name='weights_1')
weights_2 = tf.Variable(tf.random_normal([input_size_2, output_size_2]), name='weights_2')


biases_1 = tf.Variable(tf.zeros([1, output_size_1]), name='biases_1')
biases_2 = tf.Variable(tf.zeros([1, output_size_2]), name='biases_2')


layer_1 = tf.nn.relu(tf.matmul(xs, weights_1) + biases_1)
prediction = tf.matmul(layer_1, weights_2) + biases_2

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
# add variable to Graph, for reload the vairable and use
tf.add_to_collection('prediction', prediction)
tf.add_to_collection('loss', loss)



def _loss():
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		x = sess.run(xs, feed_dict={xs: x_data})
		y = sess.run(ys, feed_dict={ys: y_data})
		a = 0
		x_layer = [1,2,3,4,5,6,7,8,9,10]

		for i in range(300):
			sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

			loss2 = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
			lay1 = sess.run(layer_1, feed_dict={xs: x_data})
			plt.scatter(i, loss2, marker='*')
		plt.ion()
		plt.title('Loss Results')
		plt.xlabel('Train Steps')
		plt.ylabel('Loss value')
		plt.show()
		plt.savefig('images/loss.png', format='png')
		plt.close()
def _result_():
	saver = tf.train.Saver()
	xt = tf.placeholder(tf.float32, [None, 1])

	yt = tf.placeholder(tf.float32, [None,1])

	b1 = tf.Variable(tf.zeros([1,10]))

	b2 = tf.Variable(tf.zeros([1,1]))


	w1 = tf.Variable(tf.random_normal([1,10]))

	w2 = tf.Variable(tf.random_normal([10,1]))

	L1 = tf.nn.relu(tf.matmul(xt, w1) + b1)

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		x = sess.run(xs, feed_dict={xs: x_data})
		y = sess.run(ys, feed_dict={ys: y_data})
		a = 0

		for i in range(300):
			train_step_value, loss_value, pre = sess.run([train_step,loss,prediction], feed_dict={xs: x_data, ys: y_data})
			if i % 50 == 0:
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
				# plt.ion()
				# plt.figure(i)
				plt.subplot(2,3,a).set_title("Group{} results:".format(str(a)))
				plt.plot(x, pre, 'r')
				plt.scatter(x, y,s=2,c='b')
				plt.xlabel("x_data")
				plt.ylabel("y_data")
				
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=i)
		print(sess)
		plt.ion()
		print("Weights_1: {}, Biases_1: {}".format(sess.run(weights_1),sess.run(biases_1)))
		print("Weights_2: {}, Biases_2: {}".format(sess.run(weights_2), sess.run(biases_2)))
		plt.title("Results")
		plt.subplots_adjust(wspace=0.3, hspace=0.5)
		plt.show()
		plt.savefig('images/results.png', format='png')
		# plt.close()

_loss()
_result_()


# Results
# Weights_1: [[ 0.12498648  0.4885427   0.2738153  -0.14836657  0.15466383  0.38724378
#   -1.7780366  -1.3180089  -0.5189133   0.53980017]], 
# Biases_1: [[-0.0034627   0.00264521  0.00752168 -0.14838429  0.3298069   0.16144897
#   -0.13336018  0.13681121 -0.51895154  0.09720612]]
# Weights_2: [[ 0.08596053]
#  [-0.08175041]
#  [-1.0208174 ]
#  [-0.24553715]
#  [-1.6187183 ]
#  [ 1.1095015 ]
#  [-0.11835593]
#  [ 1.2349328 ]
#  [-0.87075645]
#  [ 1.2319158 ]],
# Biases_2: [[0.05276876]]


def loadModel():
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph('models/model.ckpt-299.meta')
		saver.restore(sess, tf.train.latest_checkpoint('models/'))
		print("Load Weights_1 from model: {}".format(sess.run('weights_1:0')))
		print("Load Weights_2 from model: {}".format(sess.run('weights_2:0')))
		print("Load Biases_1 from model: {}".format(sess.run('biases_1')))
		print("Load Biases_2 from model: {}".format(sess.run('biases_2')))



# loadModel()



	
		
			





