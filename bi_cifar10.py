import tensorflow as tf

init_kernel = tf.random_normal_initializer(mean=0, stddev=0.01)


def leakyReLu(x, alpha=0.1, name=None):
    if name:
        with tf.variable_scope(name):
            return _leakyReLu_impl(x, alpha)
    else:
        return _leakyReLu_impl(x, alpha)


def _leakyReLu_impl(x, alpha):
    return tf.nn.relu(x) - (alpha * tf.nn.relu(-x))


def discriminator(z, inp, is_training):

    # D(x)
    x = tf.reshape(inp, [-1, 32, 32, 3])

    with tf.variable_scope('disx_conv1'):
        x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')
        x = tf.layers.dropout(x, rate=0.2)
        x = tf.contrib.layers.maxout(x, num_units=2)

    with tf.variable_scope('disx_conv2'):
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='SAME')
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.contrib.layers.maxout(x, num_units=2)

    with tf.variable_scope('disx_conv3'):
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='SAME')
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.contrib.layers.maxout(x, num_units=2)

    with tf.variable_scope('disx_conv4'):
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='SAME')
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.contrib.layers.maxout(x, num_units=2)

    with tf.variable_scope('disx_conv5'):
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=1, padding='SAME')
        x = tf.layers.dropout(x, rate=0.5)
        x = tf.contrib.layers.maxout(x, num_units=2)

    # D(z) 
    # Expand input to [batch, 1, 1, channels=64]
    y = tf.expand_dims(tf.expand_dims(z, 1), 1)

     with tf.variable_scope('disz_conv1'):
        y = tf.layers.conv2d(y, filters=512, kernel_size=1, strides=1, padding='SAME')
        y = tf.layers.dropout(y, rate=0.2)
        y = tf.contrib.layers.maxout(y, num_units=2)

    with tf.variable_scope('disz_conv2'):
        y = tf.layers.conv2d(y, filters=512, kernel_size=1, strides=1, padding='SAME')
        y = tf.layers.dropout(y, rate=0.5)
        y = tf.contrib.layers.maxout(y, num_units=2)

    # D(x,z)
    #### CHECK DIMENSIONS of x and y
    d = tf.concat([x, y], axis=1)

    with tf.variable_scope('disxz_conv1'):
        d = tf.layers.conv2d(d filters=1024, kernel_size=1, strides=1, padding='SAME')
        d = tf.layers.dropout(d, rate=0.5)
        d = tf.contrib.layers.maxout(d, num_units=2)

    with tf.variable_scope('disxz_conv2'):
        d = tf.layers.conv2d(d filters=1024, kernel_size=1, strides=1, padding='SAME')
        d = tf.layers.dropout(d, rate=0.5)
        d = tf.contrib.layers.maxout(d, num_units=2)

    with tf.variable_scope('disxz_conv3'):
        d = tf.layers.conv2d(d filters=1, kernel_size=1, strides=1, padding='SAME')
        d = tf.layers.dropout(d, rate=0.5)
        logits = tf.sigmoid(d)

    return logits


def encoder(inp, is_training):
    x = tf.reshape(inp, [-1, 32, 32, 3])

    with tf.variable_scope('encoder_conv1'):
        x = tf.layers.conv2d(x, filters=32, kernel_size=5, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)

    with tf.variable_scope('encoder_conv2'):
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)

    with tf.variable_scope('encoder_conv3'):
        x = tf.layers.conv2d(x, filters=128, kernel_size=4, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)

    with tf.variable_scope('encoder_conv4'):
        x = tf.layers.conv2d(x, filters=256, kernel_size=4, strides=2, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)

    with tf.variable_scope('encoder_conv5'):
        x = tf.layers.conv2d(x, filters=512, kernel_size=4, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)

    with tf.variable_scope('encoder_dense6'):
        x = tf.layers.conv2d(x, filters=512, kernel_size=1, strides=1, padding='SAME')
        x = tf.layers.batch_normalization(x, training=is_training)
        x = leakyReLu(x)

    with tf.variable_scope('encoder_dense7'):
        x = tf.layers.conv2d(x, filters=128, kernel_size=1, strides=1, padding='SAME')

    z = tf.contrib.layers.flatten(x)

    return z


def decoder(z, is_training):

    # Expand input to [batch, 1, 1, channels=64]
    x = tf.expand_dims(tf.expand_dims(z, 1), 1)

    with tf.variable_scope('decoder_deconv_1'):
        x = tf.layers.conv2d_transpose(x, filters=256, kernel_size=4, strides=1, padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_1')
        x = leakyReLu(x)

    with tf.variable_scope('decoder_deconv_2'):
        x = tf.layers.conv2d_transpose(x, filters=128, kernel_size=4, strides=2, padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnorm_2')
        x = leakyReLu(x)

    with tf.variable_scope('decoder_deconv_3'):
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_3')
        x = leakyReLu(x)

    with tf.variable_scope('decoder_deconv_4'):
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=4, strides=2, padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_4')
        x = leakyReLu(x)

    with tf.variable_scope('decoder_deconv_5'):
        x = tf.layers.conv2d_transpose(x, filters=32, kernel_size=5, strides=1, padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_5')
        x = leakyReLu(x)

    with tf.variable_scope('decoder_dense_6'):
        x = tf.layers.conv2d(x, filters=32, kernel_size=1, strides=1, padding='SAME', kernel_initializer=init_kernel)
        x = tf.layers.batch_normalization(x, training=is_training, name='batchnormn_6')
        x = leakyReLu(x)

    with tf.variable_scope('decoder_dense_7'):
        x = tf.layers.conv2d(x, filters=3, kernel_size=1, strides=1, padding='SAME', kernel_initializer=init_kernel)
        x = tf.sigmoid(x)

    return x