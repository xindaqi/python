import numpy as np
import tensorflow as tf
import os

import matplotlib.pyplot as plt
tf.reset_default_graph()

MODEL_SAVE_PATH = "./models"
MODEL_NAME = "rnnmodel.ckpt"

'''Hidden node.'''
HIDDEN_SIZE = 30
'''Hiden layers.'''
NUM_LAYERS = 2
'''Length of rain sequence.'''
TIMESTEPS = 10
'''Train steps.'''
TRAINING_STEPS = 10000
'''bactch'''
BATCH_SIZE = 32
'''Train data numbers.'''
TRAINING_EXAMPLES = 10000
'''Test data numbers'''
TESTING_EXAMPLES = 1000
'''Sample gap.'''
SAMPLE_GAP = 0.01

def auto_number(dir_path):
    '''
    Automatic numbering the picture.

    :param dir_path: path of the image.0
    '''
    image_nums = os.listdir(dir_path)
    numbers = []
    if image_nums:
        for image_num in image_nums:
            '''extract the image name and extension name, like a.png: name:a, extension name: png'''
            number, _ = os.path.splitext(image_num)
            numbers.append(int(number))
        '''Get max number.'''
        max_number = max([x for x in numbers])
    else:
        max_number = 0
    '''Return the max number and number the image in sequence.'''
    return max_number

def generate_data(seq):
    X = []
    y = []
    for i in range(len(seq) - TIMESTEPS):
        X.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i+TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train(sess, train_X, train_y):
    '''
    Train model.

    :param sess: tensorflow session, create the work space.
    :param train_X: input data which is independent variable, shape is [?, 1, 10].
    :param train_y: input data which is dependent variable, shape is [?, 1] 
    '''

    '''Define placeholder which can get outside data from user supports.'''
    xs = tf.placeholder(tf.float32,[None, 1, 10], name="x")
    ys = tf.placeholder(tf.float32,[None, 1], name="y")
    '''Create a RNN cell composed sequentially of a number of RNNCells.'''
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
    outputs, _ = tf.nn.dynamic_rnn(cell, xs, dtype=tf.float32)
    output = outputs[:,-1,:]
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)
    loss = tf.losses.mean_squared_error(labels=ys, predictions=predictions)
        
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.train.get_global_step(),optimizer="Adagrad",
        learning_rate = 0.1
    )
    tf.add_to_collection('predictions', predictions)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    for i in range(TRAINING_STEPS):
        _, l = sess.run([train_op, loss], feed_dict={xs:train_X, ys:train_y})
        if i % 1000 == 0:
            print("train step:{}, loss: {}".format(str(i), str(l)))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
            
def run_eval(sess, test_X, test_y):
    '''
    Load trained model and predict the input data.

    :param sess: tensorflow session, create the work space.
    :param test_X: input data namely according this to predict the value compare with real value.
    :param test_y: real value will be used to compare with predict value.
    '''

    '''
    Get trained model prediction variable and use input
    data to predict output.
    '''
    pre = tf.get_collection('predictions')[0]
    graph = tf.get_default_graph()
    '''Get graph variable by name and take input data assign in it.'''
    x = graph.get_operation_by_name('x').outputs[0]
    '''Predict starting.'''
    pre_result = sess.run(pre, feed_dict={x: test_X})
    '''Auto number the figure.''' 
    numbers = auto_number("./train_result")
    '''Plot figure, show and save.'''
    plt.figure()
    plt.plot(pre_result, label='predictions', marker='x', color='r')
    plt.plot(test_y, label='real_sin', marker='|', color='b')
    plt.legend()
    plt.grid()
    plt.savefig("./train_result/{}.png".format(numbers+1),format="png")
    plt.show()
'''Test start and end node.'''
test_start = (TRAINING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
test_end = test_start+(TESTING_EXAMPLES+TIMESTEPS)*SAMPLE_GAP
'''Input data: train and test'''
train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES+TIMESTEPS, dtype=np.float32)))
'''Tensorflow session which create work space and start engine.'''
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    '''Train model.'''
    # train(sess, train_X, train_y)
    print("After RNN Training")
    '''Load trained model with meta graph, do not need redefine graph'''
    saver = tf.train.import_meta_graph('models/rnnmodel.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))
    '''Start predict.'''
    run_eval(sess, test_X, test_y)
    