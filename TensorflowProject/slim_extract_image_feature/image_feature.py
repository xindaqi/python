from __future__ import print_function
from __future__ import division
import tensorflow as tf
from nets import nets_factory
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')

# from preprocessing import preprocessing_factory
# import reader
# import model
# import time
# import losses
import utils
import os
import argparse

slim = tf.contrib.slim

# png = './train2014/test5.jpg'.lower().endswith('png')
image_path = "./images/andy002.png"
png = image_path.lower().endswith('png')

def line_draw():
    print("-----------")


def image_channel_feature():
    image = tf.read_file(image_path)
    image = tf.image.decode_png(image, channels=3) if png else tf.image.decode_jpeg(image, channels=3)
    with tf.Session() as sess:
        print("image tensor: {}".format(image))
        image_value = sess.run(image)
        channel_r = image_value[:,:,0]
        channels_num = image_value.shape[2]
        # print("Red channel: {}".format(channel_r))

        # print("image value: {}".format(image_value))
        plt.figure(figsize=(6, 6))
        for i in range(channels_num):
            channel_image = image_value[:,:,i]
            plt.subplot(3,3,i+1).set_title("图{}".format(i+1), fontproperties=font)
            plt.imshow(channel_image)
        # plt.imshow(image_value)
        # plt.imshow(channel_r)
        plt.show()


def gram(layer):
    '''
    Gram matrix funciton to extract image style.

    :params layer:image tensor.
    '''
    # print("gram layer: {}".format(layer))
    shape = tf.shape(layer)
    # print("gram shape: {}".format(shape))
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    gram_stack = tf.stack([num_images, -1, num_filters])
    # print("gram stack: {}".format(gram_stack))
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    # print("gram filters: {}".format(filters))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams


def get_style_features(FLAGS):
    """
    For the "style_image", the preprocessing step is:
    1. Resize the shorter side to FLAGS.image_size
    2. Apply central crop
    """
    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(
            FLAGS.loss_model,
            num_classes=1,
            is_training=False)
        # image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(
        #     FLAGS.loss_model,
        #     is_training=False)

        # Get the style image data
        size = FLAGS.image_size
        img_bytes = tf.read_file(FLAGS.style_image)
        # img_bytes = tf.read_file(image_path)
        if FLAGS.style_image.lower().endswith('png'):
            image = tf.image.decode_png(img_bytes, channels=3)
        else:
            image = tf.image.decode_jpeg(img_bytes, channels=3)
        # image = _aspect_preserving_resize(image, size)

        # Add the batch dimension
        # images = tf.expand_dims(image_preprocessing_fn(image, size, size), 0)
        image = tf.to_float(image)
        images = tf.expand_dims(image, 0)
        # images = tf.stack([image_preprocessing_fn(image, size, size)])

        net, endpoints_dict = network_fn(images, spatial_squeeze=False)
        # print("net: {}, endpoints_dict: {}".format(net, endpoints_dict))
        features = []
        for layer in FLAGS.style_layers:
            '''
            FLAGS.style_layers:
            vgg_16/conv1/conv1_2
            vgg_16/conv2/conv2_2
            vgg_16/conv3/conv3_3
            vgg_16/conv4/conv4_3
            '''
            feature = endpoints_dict[layer]
            '''
            image feature tensor:
            style feature: Tensor("vgg_16/conv1/conv1_2/Relu:0", shape=(1, ?, ?, 64), dtype=float32)
            style feature: Tensor("vgg_16/conv2/conv2_2/Relu:0", shape=(1, ?, ?, 128), dtype=float32)
            style feature: Tensor("vgg_16/conv3/conv3_3/Relu:0", shape=(1, ?, ?, 256), dtype=float32)
            style feature: Tensor("vgg_16/conv4/conv4_3/Relu:0", shape=(1, ?, ?, 512), dtype=float32)
            '''
            print("style feature: {}".format(feature))
            line_draw()
            '''remove extra batch dimension and get source image feature extracted by NN.'''
            feature = tf.squeeze(feature, [0])
            '''Gram matrix extract image style.'''
            # feature = tf.squeeze(gram(feature), [0])

            print("gram cal in losses: {}".format(feature))
            line_draw()
            features.append(feature)

        with tf.Session() as sess:
            '''Restore variables for loss network.'''
            init_func = utils._get_init_fn(FLAGS)
            init_func(sess)
            # Make sure the 'generated' directory is exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            # Indicate cropped style image path
            save_file = 'generated/target_style_' + FLAGS.naming + '.jpg'
            # Write preprocessed style image to indicated path
            with open(save_file, 'wb') as f:
                target_image = images[0, :]
                value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                f.write(sess.run(value))
                tf.logging.info('Target style pattern is saved to: %s.' % save_file)

            '''Return the features those layers are use for measuring style loss.'''
            return sess.run(features)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', default='conf/mosaic.yml', help='the path to the conf file')
    return parser.parse_args()
def line_draw():
    print("-------------")

def main(FLAGS):
    '''processing target style image'''
    # style_features_t = losses.get_style_features(FLAGS)
    style_features_t = get_style_features(FLAGS)
    '''lenght of style feature: 4'''
    print("lenght of style feature: {}".format(len(style_features_t)))
    print("shape of style feature: {}".format(style_features_t[0].shape))
    '''Image channels nmuber:[64, 128, 256, 512]'''
    channels_num = [images.shape[2] for images in style_features_t]
    '''Neural network layer which extract image feature.'''
    feature_name = ["VGG16卷积神经网络conv1/conv1_2层特征","VGG16卷积神经网络/conv2/conv2_2层特征",
                    "VGG16卷积神经网络conv3/conv3_3层特征","VGG16卷积神经网络conv4/conv4_3层特征"]
    '''Saved image name.'''
    save_image_name = ["conv1_conv1_2", "/conv2_conv2_2", "conv3_conv3_3", "conv4_conv4_3"]
    '''Image column number extract from NN.'''
    column_size = [8, 16, 32, 64]
    '''Figure width size.'''
    figure_width = [10, 15, 25, 30]
    '''[64, 128, 256, 512]'''
    print("channels number list: {}".format(channels_num))
    for i in range(len(channels_num)):
        '''
        Plot image feature.
        :parmas i:get corresponding value from up list, e.g.channels_num, feature_name.
        :parms j:channels number.
        '''
        plt.figure(figsize=(figure_width[i], 10))
        plt.suptitle(feature_name[i], fontproperties=font, x=0.5, y=0.95, fontsize=10)
        image_num = channels_num[i]
        for j in range(image_num):
            plt.subplot(8,column_size[i],j+1).set_title("图{}".format(j+1), fontproperties=font)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.5, hspace=0.5)
            plt.imshow(style_features_t[i][:,:,j])
            plt.axis("off")
            plt.colorbar()
        plt.savefig("./image_features/{}_style.png".format(save_image_name[i]), format="png")
        plt.show()

    # print("style features: {}".format(style_features_t))

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()
    FLAGS = utils.read_conf_file(args.conf)
    main(FLAGS)
    # image_channel_feature()
