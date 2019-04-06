import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

mb_size = 32
'''X_dim: Real image for Discriminator.'''
X_dim = 784
'''noise_dim: generate image's dimension.'''
noise_dim = 64
'''hidden dim: hidden layer dimension.'''
hidden_dim = 128
'''lr: learning rate.'''
lr = 1e-3
d_steps = 3

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
# print("train data numbers: {}".format(mnist.train.num_examples))
def extract_data():
    images = mnist.train.images
    labels = mnist.train.labels
    return images, labels

def plot(samples):
    '''Plot generate image.
    :params samples: generate image matrix.
    return:
    fig: figure box object.
    '''
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        '''cmap: for grey image or it will be change color'''
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

def xavier_init(size):
    '''Initializer the weights and biases.
    :params size: data shape.
    return:
    random data with specify shape.
    '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def log(x):
    '''Calculate data with log methods.
    :parmas x: input data.
    return:
    cal results.
    '''
    return tf.log(x + 1e-8)

'''Real image input.'''
X = tf.placeholder(tf.float32, shape=[None, X_dim])
'''Generator image input.'''
z = tf.placeholder(tf.float32, shape=[None, noise_dim])
'''Discriminator network paremeters.'''
D_W1 = tf.Variable(xavier_init([X_dim + noise_dim, hidden_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))
D_W2 = tf.Variable(xavier_init([hidden_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))
'''Process real image network parements.'''
Q_W1 = tf.Variable(xavier_init([X_dim, hidden_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))
Q_W2 = tf.Variable(xavier_init([hidden_dim, noise_dim]))
Q_b2 = tf.Variable(tf.zeros(shape=[noise_dim]))
'''Generator network parements.'''
P_W1 = tf.Variable(xavier_init([noise_dim, hidden_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[hidden_dim]))
P_W2 = tf.Variable(xavier_init([hidden_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
'''Variable lists.'''
theta_G = [Q_W1, Q_W2, Q_b1, Q_b2, P_W1, P_W2, P_b1, P_b2]
theta_D = [D_W1, D_W2, D_b1, D_b2]

def sample_z(m, n):
    '''Generate noise data for test Genreate network.
    :parmas m: row size
    :params n: column size
    return:
    random data
    '''
    return np.random.uniform(-1., 1., size=[m, n])

def process_real_image(X):
    '''Process real images
    :params X: image matrix
    return:
    output image shape: batch*64
    '''
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    h = tf.matmul(h, Q_W2) + Q_b2
    '''batch x 64'''
    return h

def generate_image(z):
    '''Generator fake image and process with sigmoid method.
    :params z: input noise image matrix
    return:
    sigmoid data
    '''
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    h = tf.matmul(h, P_W2) + P_b2
    '''batch x 784'''
    return tf.nn.sigmoid(h)


def discriminate_image(X, z):
    '''Discriminate input image whether real or not.
    :parmas X: real image
    :params z: noise image
    return:
    sigmoid data
    '''
    inputs = tf.concat([X, z], axis=1)
    h = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    '''batch x 1'''
    return tf.nn.sigmoid(tf.matmul(h, D_W2) + D_b2)

'''Create batch*64 image.'''
z_hat = process_real_image(X)
'''Create batch*784 image.'''
X_hat = generate_image(z)
'''Discrimiate real image and processed image.'''
D_enc = discriminate_image(X, z_hat)
'''Discriminate processed and generate image.'''
D_gen = discriminate_image(X_hat, z)
'''Discriminate network loss.'''
D_loss = -tf.reduce_mean(log(D_enc) + log(1 - D_gen))
'''Genreate network loss.'''
G_loss = -tf.reduce_mean(log(D_gen) + log(1 - D_enc))
'''Optimize discriminate network loss.'''
D_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(D_loss, var_list=theta_D))
'''Optimize generate network loss.'''
G_solver = (tf.train.AdamOptimizer(learning_rate=lr)
            .minimize(G_loss, var_list=theta_G))

# sess = tf.Session()
def train():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        if not os.path.exists('out/'):
            os.makedirs('out/')
        if not os.path.exists('models/'):
            os.makedirs("models/")
        i = 0
        for it in range(10001):
            '''Extract image data from datasets.'''
            X_mb, _ = mnist.train.next_batch(mb_size)
            # print("image data: {}".format(X_mb))
            '''Generate noise.'''
            z_mb = sample_z(mb_size, noise_dim)

            _, D_loss_curr = sess.run(
                [D_solver, D_loss], feed_dict={X: X_mb, z: z_mb}
            )

            _, G_loss_curr = sess.run(
                [G_solver, G_loss], feed_dict={X: X_mb, z: z_mb}
            )

            if it % 1000 == 0:
                print('Iter: {}; D_loss: {:.4}; G_loss: {:.4}'
                      .format(it, D_loss_curr, G_loss_curr))

                samples = sess.run(X_hat, feed_dict={z: sample_z(16, noise_dim)})

                '''Save evaluate results.'''
                fig = plot(samples)
                plt.savefig('out/{}.png'
                            .format(str(i).zfill(3)), bbox_inches='tight')
                i += 1
                plt.close(fig)
            saver.save(sess, "./models/gan_test.ckpt")

def line_draw():
    print("------------------------")
if __name__ == "__main__":
    train()
    # images, labels = extract_data()
    # print("Train image: {}".format(images))
    # print("images number: {}".format(len(images)))
    # line_draw()
    # print("Train labels: {}".format(labels))
    # # print("image value: {}".format(images[0]))
    # print("image shape: {}".format(images[0].shape))
    # print("type of image: {}".format(type(images[0])))
    # image_string = images[0].tostring()
    # print("image data to string: {}".format(image_string))
    # print("image type: {}".format(type(image_string)))
    # image_decode = tf.decode_raw(image_string, tf.uint8)
    # print("image decode shape: {}".format(image_decode.shape))
    # image_reshape = tf.reshape(image_decode, [28, 28, 1])
    # print("image reshape: {}".format(image_reshape.shape))
    # # image_decode = tf.to_float(image_decode)
    # # plt.figure(figsize=(6, 6))
    # # plt.imshow(image_decode)
    # # plt.show()
