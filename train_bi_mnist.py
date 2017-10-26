import os
import time
import numpy as np
import tensorflow as tf
import bi_mnist
import sys


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('seed', 146, 'seed')
flags.DEFINE_integer('seed_data', 646, 'seed data')
flags.DEFINE_integer('freq_print', 20, 'print frequency image tensorboard [20]')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate dis[0.0003]')


FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")

def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)
    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling
    print('loading data')
    data = np.load('./mnist.npz')
    trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(np.float32)
    trainx_2 = trainx.copy()
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)

    '''//////construct graph //////'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='unlabeled_data_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')

    gen = bi_mnist.decoder
    enc = bi_mnist.encoder
    dis = bi_mnist.discriminator

    with tf.variable_scope('encoder_model'):
        z_gen = enc(inp, is_training=is_training_pl)

    with tf.variable_scope('generator_model') as scope:
        z = tf.random_uniform([FLAGS.batch_size, 256])
        x_gen = gen(z, is_training=is_training_pl)
        scope.reuse_variables()
        reconstruct = gen(z_gen, is_training=is_training_pl)  # reconstruction image dataset though bottleneck

    with tf.variable_scope('discriminator_model') as scope:
        l_encoder = dis(z_gen,inp, is_training=is_training_pl)
        scope.reuse_variables()
        l_generator = dis(z,x_gen, is_training=is_training_pl)


    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder))
        loss_dis_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator))
        loss_discriminator = loss_dis_gen + loss_dis_enc
        # generator
        loss_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator),logits=l_generator))
        # encoder
        loss_encoder = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder),logits=l_encoder))


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_enc = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen): # attached op for moving average batch norm
            train_gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)
        with tf.control_dependencies(update_ops_enc):
            train_enc_op = optimizer_enc.minimize(loss_encoder, var_list=evars)
        with tf.control_dependencies(update_ops_dis):
            train_dis_op = optimizer_dis.minimize(loss_discriminator, var_list=dvars)

    with tf.name_scope('summary'):
        with tf.name_scope('dis_summary'):
            tf.summary.scalar('loss_discriminator', loss_discriminator, ['dis'])
            tf.summary.scalar('loss_dis_encoder', loss_dis_enc, ['dis'])
            tf.summary.scalar('loss_dis_gen', loss_dis_gen, ['dis'])

        with tf.name_scope('gen_summary'):
            tf.summary.scalar('loss_generator', loss_generator, ['gen'])
            tf.summary.scalar('loss_encoder', loss_encoder, ['gen'])

        with tf.name_scope('image_summary'):
            tf.summary.image('reconstruct', reconstruct, 10, ['image'])
            tf.summary.image('input_images', tf.reshape(inp, [-1,28,28,1]), 10, ['image'])

        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')


    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        print('initialization done')

        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

        train_batch = 0

        for epoch in range(200):
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = trainx[rng.permutation(trainx.shape[0])]  # shuffling unl dataset
            trainx_2 = trainx_2[rng.permutation(trainx.shape[0])]

            train_loss_dis, train_loss_gen, train_loss_enc = [ 0, 0, 0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t,nr_batches_train)
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {inp: trainx[ran_from:ran_to],is_training_pl: True}
                _, ld, sm = sess.run([train_dis_op, loss_discriminator, sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                writer.add_summary(sm, train_batch)

                # train generator and encoder
                feed_dict = {inp: trainx_2[ran_from:ran_to],is_training_pl: True}
                _,_, lg, le, sm = sess.run([train_gen_op, train_enc_op, loss_encoder, loss_generator, sum_op_gen], feed_dict=feed_dict)
                train_loss_gen += lg
                train_loss_enc += le
                writer.add_summary(sm, train_batch)

                if t % FLAGS.freq_print == 0:  # inspect reconstruction
                    t= np.random.randint(0,4000)
                    ran_from = t
                    ran_to = t + FLAGS.batch_size
                    sm = sess.run(sum_op_im, feed_dict={inp: trainx[ran_from:ran_to],is_training_pl: False})
                    writer.add_summary(sm, train_batch)

                train_batch += 1

            train_loss_gen /= nr_batches_train
            train_loss_enc /= nr_batches_train
            train_loss_dis /= nr_batches_train


            print("Epoch %d | time = %ds | loss gen = %.4f | loss enc = %.4f | loss dis = %.4f "
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_enc, train_loss_dis))

if __name__ == '__main__':
    tf.app.run()
