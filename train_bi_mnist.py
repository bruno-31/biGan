import os
import time
import numpy as np
import tensorflow as tf
import mnist_gan

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log', 'log directory')
flags.DEFINE_integer('seed', 146, 'seed')
flags.DEFINE_integer('seed_data', 646, 'seed data')
flags.DEFINE_integer('seed_tf', 646, 'tf random seed')
flags.DEFINE_integer('labeled', 10, 'labeled image per class[100]')
flags.DEFINE_float('learning_rate_d', 0.003, 'learning_rate dis[0.003]')
flags.DEFINE_float('learning_rate_g', 0.003, 'learning_rate gen[0.003]')
flags.DEFINE_float('unl_weight', 1, 'unlabeled weight [1.]')
flags.DEFINE_float('lbl_weight', 1, 'labeled weight [1.]')
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

    # select labeled data
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    print('labeled digits : ', len(tys))


    '''construct graph'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='labeled_data_input_pl')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='unlabeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')

    gen = mnist_gan.generator
    dis = mnist_gan.discriminator

    with tf.variable_scope('generator_model'):
        gen_inp = gen(batch_size=FLAGS.batch_size, is_training=is_training_pl)

    with tf.variable_scope('discriminator_model') as scope:
        init_weight_op, _ = dis(inp, is_training_pl, True)
        scope.reuse_variables()
        logits_lab, _ = dis(inp, is_training_pl)
        logits_unl, layer_real = dis(unl, is_training_pl)
        logits_gen, layer_fake = dis(gen_inp, is_training_pl)

    with tf.variable_scope("model_test") as test_scope:
        _,_ = dis(inp, is_training_pl,True)
        test_scope.reuse_variables()
        logits_test, _ = dis(inp, is_training_pl, False)

    with tf.name_scope('loss_functions'):
        # Improved gan, T. Salimans
        # DISCRIMINATOR
        z_exp_lab = tf.reduce_mean(tf.reduce_logsumexp(logits_lab, axis=1))
        rg = tf.cast(tf.range(0,FLAGS.batch_size), tf.int32)
        idx = tf.stack([rg,lbl], axis=1)
        l_lab = tf.gather_nd(logits_lab, idx)
        loss_lab = -tf.reduce_mean(l_lab)+z_exp_lab

        l_unl = tf.reduce_logsumexp(logits_unl,axis=1)
        d = tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(logits_unl,axis=1)), axis=0) + \
            tf.reduce_mean(tf.nn.softplus(tf.reduce_logsumexp(logits_gen,axis=1)), axis=0)
        loss_unl = -0.5*tf.reduce_mean(l_unl) + 0.5*d

        loss_dis = FLAGS.unl_weight * loss_unl + FLAGS.lbl_weight * loss_lab

        accuracy_dis_unl = tf.reduce_mean(tf.cast(tf.greater(logits_unl, 0), tf.float32))
        accuracy_dis_gen = tf.reduce_mean(tf.cast(tf.less(logits_gen, 0), tf.float32))
        accuracy_dis = 0.5 * accuracy_dis_unl + 0.5* accuracy_dis_gen

        correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        correct_pred_test = tf.equal(tf.cast(tf.argmax(logits_test, 1), tf.int32), tf.cast(lbl, tf.int32))
        accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))
        # GENERATOR
        m1 = tf.reduce_mean(layer_real, axis=0)
        m2 = tf.reduce_mean(layer_fake, axis=0)
        loss_gen =  tf.reduce_mean(tf.square(m1-m2))
        fool_rate = tf.reduce_mean(tf.cast(tf.greater(logits_gen, 0), tf.float32))

    with tf.name_scope('optimizers'):
        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        testvars = [var for var in tvars if 'model_test' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]

        optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_d, beta1=0.5, name='dis_optimizer')
        optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_g, beta1=0.5, name='gen_optimizer')

        train_dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)
        ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        maintain_averages_op = ema.apply(dvars)

        with tf.control_dependencies([train_dis_op]):
                training_op = tf.group(maintain_averages_op)

        copy_graph = [tf.assign(x,ema.average(y)) for x,y in zip(testvars,dvars)]

        with tf.control_dependencies(update_ops_gen):
            train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)


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
