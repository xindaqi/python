import tensorflow as tf
import yaml

slim = tf.contrib.slim


def _get_init_fn(FLAGS):
    """
    This function is copied from TF slim.

    Returns a function run by the chief worker to warm-start the training.

    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    tf.logging.info('Use pretrained model %s' % FLAGS.loss_model_file)

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    '''exclusions: ['vgg_16/fc']'''
    # print("exclusions: {}".format(exclusions))
    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        '''
        variable name: <tf.Variable 'vgg_16/conv1/conv1_1/weights:0' shape=(3, 3, 3, 64) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv1/conv1_1/biases:0' shape=(64,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv1/conv1_2/weights:0' shape=(3, 3, 64, 64) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv1/conv1_2/biases:0' shape=(64,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv2/conv2_1/weights:0' shape=(3, 3, 64, 128) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv2/conv2_1/biases:0' shape=(128,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv2/conv2_2/weights:0' shape=(3, 3, 128, 128) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv2/conv2_2/biases:0' shape=(128,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv3/conv3_1/weights:0' shape=(3, 3, 128, 256) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv3/conv3_1/biases:0' shape=(256,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv3/conv3_2/weights:0' shape=(3, 3, 256, 256) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv3/conv3_2/biases:0' shape=(256,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv3/conv3_3/weights:0' shape=(3, 3, 256, 256) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv3/conv3_3/biases:0' shape=(256,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv4/conv4_1/weights:0' shape=(3, 3, 256, 512) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv4/conv4_1/biases:0' shape=(512,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv4/conv4_2/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv4/conv4_2/biases:0' shape=(512,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv4/conv4_3/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv4/conv4_3/biases:0' shape=(512,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv5/conv5_1/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv5/conv5_1/biases:0' shape=(512,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv5/conv5_2/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv5/conv5_2/biases:0' shape=(512,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv5/conv5_3/weights:0' shape=(3, 3, 512, 512) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/conv5/conv5_3/biases:0' shape=(512,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/fc6/weights:0' shape=(7, 7, 512, 4096) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/fc6/biases:0' shape=(4096,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/fc7/weights:0' shape=(1, 1, 4096, 4096) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/fc7/biases:0' shape=(4096,) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/fc8/weights:0' shape=(1, 1, 4096, 1) dtype=float32_ref>
        variable name: <tf.Variable 'vgg_16/fc8/biases:0' shape=(1,) dtype=float32_ref>
        '''
        # print("variable name: {}".format(var))
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        FLAGS.loss_model_file,
        variables_to_restore,
        ignore_missing_vars=True)
    # return "test"


class Flag(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def read_conf_file(conf_file):
    with open(conf_file) as f:
        FLAGS = Flag(**yaml.load(f))
    return FLAGS


def mean_image_subtraction(image, means):
    image = tf.to_float(image)

    num_channels = 3
    channels = tf.split(image, num_channels, 2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, 2)


if __name__ == '__main__':
    f = read_conf_file('conf/mosaic.yml')
    print(f.loss_model_file)
