import tensorflow as tf 
import time

# Choice GPU:0
with tf.device("/gpu:0"):
	v1 = tf.constant([125.0], shape=[1], name='v1')
	v2 = tf.constant([125.0], shape=[1], name='v2')
	startTime = time.time()
	result = v1 + v2
	endTime = time.time()
	# Calculate time consuming
	timeConsuming = endTime - startTime
	# Display device Info.
	config = tf.ConfigProto(log_device_placement=True)
	with tf.Session(config=config) as sess:
		print("GPU calculator result is : {}".format(sess.run(result)))
		print("GPU calculator time consuming is: {}".format(timeConsuming))
