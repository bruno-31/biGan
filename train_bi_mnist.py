import os
import time
import numpy as np
import tensorflow as tf
import bi_mnist

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('seed', 146, 'seed')
flags.DEFINE_integer('seed_data', 646, 'seed data')
flags.DEFINE_integer('seed_tf', 646, 'tf random seed')
flags.DEFINE_integer('labeled', 10, 'labeled image per class[100]')
flags.DEFINE_float('learning_rate', 0.0003, 'learning_rate dis[0.0003]')


FREQ_PRINT = 1000
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.lower(), value))
print("")

def main(_):
    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling
    tf.set_random_seed(FLAGS.seed_tf)
    print('loading data')
    # load MNIST data
    data = np.load('../data/mnist.npz')
    trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(np.float32)
    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    testx = data['x_test'].astype(np.float32)
    testy = data['y_test'].astype(np.int32)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)


    '''construct graph'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='unlabeled_data_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')

    gen = bi_mnist.generator
    enc = bi_mnist.decoder
    dis = bi_mnist.discriminator

    with tf.variable_scope('generator_model'):
        z = tf.random_uniform([128])
        x_gen = gen(z, is_training=is_training_pl)

    with tf.variable_scope('encoder_model'):
        z_gen = enc(inp, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model') as scope:
        l_encoder = dis(z_gen,inp, is_training=is_training_pl)
        scope.reuse_variables()
        l_generator = dis(z,x_gen, is_training=is_training_pl)


    with tf.name_scope('loss_functions'):
        # discriminator
        loss_dis_enc = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_encoder),logits=l_encoder)
        loss_dis_gen = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_generator),logits=l_generator)
        loss_discriminator = loss_dis_gen + loss_dis_enc

        # generator
        loss_generator = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(l_generator),logits=l_generator)

        # encoder
        loss_generator = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(l_encoder),logits=l_encoder)


    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        evars = [var for var in tvars if 'encoder_model' in var.name]

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_en = [x for x in update_ops if ('encoder_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, name='gen_optimizer')
        optimizer_enc = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, beta1=0.5, name='enc_optimizer')

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_generator, var_list=gvars)

        with tf.control_dependencies(update_ops_enc):
            train_enc_op = optimizer_gen.minimize(loss_generator, var_list=evars)

        with tf.control_dependencies(update_ops_dis):
            train_dis_op = optimizer_gen.minimize(loss_generator, var_list=dvars)



    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        #Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        sess.run(init, feed_dict={inp: trainx_unl[0:FLAGS.batch_size], is_training_pl: True})
        print('initialization done')

        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        train_batch = 0
        for epoch in range(200):
            begin = time.time()

            # construct randomly permuted minibatches
            trainx = []
            trainy = []
            for t in range(int(np.ceil(trainx_unl.shape[0] / float(txs.shape[0])))):  # same size lbl and unlb
                inds = rng.permutation(txs.shape[0])
                trainx.append(txs[inds])
                trainy.append(tys[inds])
            trainx = np.concatenate(trainx, axis=0)
            trainy = np.concatenate(trainy, axis=0)
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  # shuffling unl dataset
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            train_loss_lab, train_loss_unl, train_loss_gen, train_acc, test_acc = [ 0, 0, 0, 0, 0]
            # training
            for t in range(nr_batches_train):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # train discriminator
                feed_dict = {inp: trainx[ran_from:ran_to],
                             lbl: trainy[ran_from:ran_to],
                             unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True}
                _, ll, lu, acc, sm = sess.run([training_op, loss_lab, loss_unl, accuracy, sum_op_dis],
                                              feed_dict=feed_dict)
                train_loss_lab += ll
                train_loss_unl += lu
                train_acc += acc
                writer.add_summary(sm, train_batch)

                # train generator
                _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                                                                                      is_training_pl: True})
                train_loss_gen += lg
                writer.add_summary(sm, train_batch)

                if t % FREQ_PRINT == 0:
                    sm = sess.run(sum_op_im, feed_dict={is_training_pl: False})
                    writer.add_summary(sm, train_batch)
                train_batch += 1
            train_loss_lab /= nr_batches_train
            train_loss_unl /= nr_batches_train
            train_acc /= nr_batches_train

            # Testing
            sess.run(copy_graph)
            for t in range(nr_batches_test):
                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size
                feed_dict = {inp: testx[ran_from:ran_to],
                             lbl: testy[ran_from:ran_to],
                             is_training_pl: False}
                test_acc += sess.run(accuracy_test, feed_dict=feed_dict)

            test_acc /= nr_batches_test

            # Plotting
            sum = sess.run(sum_op_epoch, feed_dict={acc_train_pl: train_acc,
                                                    acc_test_pl: test_acc})
            writer.add_summary(sum, epoch)

            print("Epoch %d--Time = %ds | loss gen = %.4f | loss lab = %.4f | loss unl = %.4f "
                  "| train acc = %.4f| test acc = %.4f"
                  % (epoch, time.time() - begin, train_loss_gen, train_loss_lab, train_loss_unl, train_acc, test_acc))

if __name__ == '__main__':
    tf.app.run()
