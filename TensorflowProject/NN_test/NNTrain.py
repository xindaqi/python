import tensorflow as tf 
import numpy as np 
from NNInference import *



def train(x, y):
	with tf.name_scope("Input"):
		xs = tf.placeholder(tf.float32, [None, 1], name='x')
		ys = tf.placeholder(tf.float32, [None, 1], name='y')

	prediction = inference(xs)

	with tf.name_scope("Loss"):
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
		
	with tf.name_scope("Train_Step"):
		train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		a = 0

		for i in range(300):
			train_step_value, loss_value, pre = sess.run([train_step,loss,prediction], feed_dict={xs: x, ys: y})
			if i % 50 == 0:
				a += 1
				# w1 = sess.run(weights_1)
				# w2 = sess.run(weights_2)
				# print("Weights_1 :{}".format(w1))
				# print("weights_2 :{}".format(w2))
				# print(a)
				# loss_1 = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
				print("Loss :{}".format(loss_value))
				print(pre)
				# print(loss)
				# print(train_step_value)
				# plt.ion()
				# plt.figure(i)
				plt.subplot(2,3,a).set_title("Group{} results:".format(str(a)))
				plt.plot(x, pre, 'r')
				plt.scatter(x, y,s=2,c='b')
				plt.xlabel("x_data")
				plt.ylabel("y_data")				
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=i)
		# print(sess)
		# plt.ion()
		# print("Weights_1: {}, Biases_1: {}".format(sess.run(weights_1),sess.run(biases_1)))
		# print("Weights_2: {}, Biases_2: {}".format(sess.run(weights_2), sess.run(biases_2)))
		plt.title("Results")
		plt.subplots_adjust(wspace=0.3, hspace=0.5)
		plt.show()
		# plt.savefig('images/results.png', format='png')
		# plt.close()
		writer = tf.summary.FileWriter(LOG_DIR, tf.get_default_graph())
		writer.close()

def main(argv=None):
	x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis] 
	noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
	y_data = np.square(x_data) - 0.5*x_data + noise
	
	train(x_data, y_data)

if __name__ == '__main__':
	tf.app.run()