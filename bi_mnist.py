import tensorflow as tf
import nn

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)

def gaussian_noise_layer(input_layer, std, deterministic):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    # if deterministic or std==0:
    #     return input_layer
    # else:
    #     return input_layer + noise
    y= tf.cond(deterministic, lambda :input_layer, lambda :input_layer+noise)
    return y

def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)

def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def discriminator(inp,z, is_training):
    #D(x)
    x=tf.layers.conv2d(inp, 32,[3,3],padding='SAME')

    x=tf.layers.conv2d(x, 64,[3,3],padding='SAME',strides=[2,2]) #14*14
    x=tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)
    x=tf.layers.dropout(x,rate=0.2)

    x=tf.layers.conv2d(x, 128,[3,3],padding='SAME', strides=[2,2]) #7*7
    x=tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)
    x=tf.layers.dropout(x,rate=0.2)

    #D(z)
    y = tf.layers.conv2d(z, 128, [1, 1], padding='SAME')
    y = leakyReLu(y)
    y =tf.layers.dropout(y,rate=0.2)
    y = tf.layers.conv2d(y, 128, [1, 1], padding='SAME')
    y = leakyReLu(y)
    y = tf.layers.dropout(y, rate=0.2)

    #D(x,z)
    x = tf.concat([x,y], axis=3)

    x = tf.layers.conv2d(x, 512, [1, 1], padding='SAME')
    x = leakyReLu(x)
    x = tf.layers.dropout(x, rate=0.2)

    x = tf.layers.conv2d(x, 512, [1, 1], padding='SAME')
    x = leakyReLu(x)
    x = tf.layers.dropout(x, rate=0.2)

    logits = tf.layers.conv2d(x, 1, [1, 1], padding='SAME')

    return logits


def encoder(inp, is_training):
    # D(x)
    x = tf.layers.conv2d(inp, 32, [3, 3], padding='SAME')

    x = tf.layers.conv2d(x, 64, [3, 3], padding='SAME', strides=[2, 2])
    x = tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)

    x = tf.layers.conv2d(x, 128, [3, 3], padding='SAME', strides=[2, 2])
    x = tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)

    z = tf.layers.conv2d(x, 512, [1, 1], padding='SAME', strides=[1, 1])
    return z

def decoder(z, batch_size,is_training):

    x = tf.layers.conv2d_transpose(z,128,[4,4])
    x = tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)

    x = tf.layers.conv2d_transpose(z, 64, [4, 4], strides=[2,2])
    x = tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)

    x = tf.layers.conv2d_transpose(z, 32, [4, 4])
    x = tf.layers.batch_normalization(x, is_training)
    x = leakyReLu(x)

    x = tf.layers.conv2d(x, 3, [1, 1], padding='SAME', strides=[1, 1])
    x = tf.sigmoid(x)
    return x
