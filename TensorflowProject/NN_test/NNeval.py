import tensorflow as tf 
import time

from NNInference import *

EVAL_INTERVAL_SECS = 10

def test(xs, ys):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf.float32, [None, 1], name='x-input')
		y_ = tf.placeholder(tf.float32, [None, 1], name='y-input')

		test_feed = {x: xs, y_: ys}

		pre = inference(x)
		loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-pre), reduction_indices=[1]))
		saver = tf.train.Saver()
		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

				global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
				pre, loss = sess.run([pre, loss], feed_dict=test_feed)
				print("After {} training steps, loss is: {}".format(global_step, loss))

				plt.plot(xs, pre, 'r')
				plt.scatter(xs, ys, s=2, c='b')
				plt.show()
			else:
				print("No checkpoint file!")
				return 

def main(argv=None):
	x_data = np.linspace(-1, 1, 250, dtype=np.float32)[:, np.newaxis] 
	noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
	y_data = np.square(x_data) - 0.5*x_data + noise
	
	test(x_data, y_data)

if __name__ == '__main__':
	tf.app.run()