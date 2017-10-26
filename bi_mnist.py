import tensorflow as tf

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.05)


def gaussian_noise_layer(input_layer, std, deterministic):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    # if deterministic or std==0:
    #     return input_layer
    # else:
    #     return input_layer + noise
    y = tf.cond(deterministic, lambda: input_layer, lambda: input_layer + noise)
    return y


def leakyReLu(x, alpha=0.2, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def discriminator(z, inp, is_training):
    # D(x)
    x = tf.reshape(inp, [-1, 28, 28, 1])
    with tf.variable_scope('layer_1'):
        x = tf.layers.conv2d(x, 32, [3, 3], padding='SAME')

    with tf.variable_scope('layer_2'):
        x = tf.layers.conv2d(x, 64, [3, 3], padding='SAME', strides=[2, 2])  # 14*14*1
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)
        x = tf.layers.dropout(x, rate=0.5)

    with tf.variable_scope('layer_3'):
        x = tf.layers.conv2d(x, 128, [3, 3], padding='SAME', strides=[2, 2])  # 7*7*1
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)
        x = tf.layers.max_pooling2d(x, pool_size=[2, 2], strides=[2,2])  # 3*3
        x = tf.layers.dropout(x, rate=0.5)

    x=tf.reshape(x, [-1, 3 * 3 * 128])
    # D(z) z 256
    with tf.variable_scope('layer_4'):
        y = tf.layers.dense(z, 128)
        y = leakyReLu(y)
        y = tf.layers.dropout(y, rate=0.5)

    with tf.variable_scope('layer_5'):
        y = tf.layers.dense(z, 256)
        y = leakyReLu(y)
        y = tf.layers.dropout(y, rate=0.5)

    # D(x,z)
    x = tf.concat([x, y], axis=1)
    with tf.variable_scope('layer_6'):
        x = tf.layers.dense(x, 512)
        x = leakyReLu(x)
        x = tf.layers.dropout(x, rate=0.5)

    with tf.variable_scope('layer_8'):
        logits = tf.layers.dense(x, 1)

    return logits


def encoder(inp, is_training):
    x = tf.reshape(inp, [-1, 28, 28, 1])

    x = tf.layers.conv2d(x, 32, [3, 3], padding='SAME')

    x = tf.layers.conv2d(x, 64, [3, 3], padding='SAME', strides=[2, 2])
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)

    x = tf.layers.conv2d(x, 128, [3, 3], padding='SAME', strides=[2, 2])
    x = tf.layers.batch_normalization(x, training=is_training)
    x = leakyReLu(x)

    x = tf.contrib.layers.flatten(x)

    z = tf.layers.dense(x, 256)
    return z


def decoder(z, is_training):
    with tf.variable_scope('dense_1'):
        x = tf.layers.dense(z, units=3 *3 * 256, kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
        x = tf.nn.relu(x)

    x = tf.reshape(x, [-1, 3, 3, 256])

    with tf.variable_scope('deconv_1'):
        x = tf.layers.conv2d_transpose(x, 256, [3, 3], strides=[2, 2], padding='VALID', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
        x = tf.nn.relu(x)

    with tf.variable_scope('deconv_2'):
        x = tf.layers.conv2d_transpose(x, 128, [5, 5], strides=[2, 2], padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
        x = tf.nn.relu(x)
    # including weightnorm     # [batch,32,32,3]
    with tf.variable_scope('deconv_3'):
        x = tf.layers.conv2d_transpose(x, 1, [5, 5], strides=[2, 2], padding='SAME', kernel_initializer=init_kernel,
                                       activation=tf.sigmoid)
    return x
