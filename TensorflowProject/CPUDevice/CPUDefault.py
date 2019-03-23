import tensorflow as tf 
import time

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
	print("CPU calculator result is : {}".format(sess.run(result)))
	print("CPU calculator time consuming is: {}".format(timeConsuming))