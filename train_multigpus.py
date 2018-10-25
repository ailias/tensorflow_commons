import tensorflow as tf
import argparse
import os
import numpy as np
import time
from model as Model
from dataset import batch_data

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, default='0,2,3')
parser.add_argument("--data_path", type=str, default='/path/to/record/folder')
parser.add_argument("--output_path", type=str, default='Data/output')
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--max_steps", type=int, default=10e6)
parser.add_argument("--decay_steps", type=int, default=30000)
parser.add_argument("--decay_rate", type=float, default=0.8)
parser.add_argument("--display_steps", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-3)


def average_gradients(tower_grads):
    r'''
    Do all gradients average of each device batch data.
    :param tower_grads:
    :return:  averaged grad_and_var array
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def calc_loss(criterion, outputs, labels, mask=None):
    r'''
    Calculate loss of model output
    :param criterion:
    :param outputs:
    :param labels:
    :param mask:
    :return:
    '''

    loss = criterion(outputs, labels, weights=mask)
    return loss


def train(model, optimizer, batch_inputs, batch_labels, batch_mask):
    r'''
    Compute loss for each device batch data
    :param model:
    :param optimizer:
    :param batch_inputs:
    :param batch_labels:
    :param batch_mask:
    :return:
    '''
    # model inference
    outputs = model(batch_inputs, is_training=True)

    # model loss
    criterion = tf.losses.absolute_difference
    loss = calc_loss(criterion, batch_labels, outputs, batch_mask * 16)

    # model grad
    grad = optimizer.compute_gradients(loss)

    return grad, outputs, loss


def main(args):
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # set network model
        model = Model()

        # set global_step and lr
        global_step = tf.train.get_or_create_global_step()
        lr = tf.train.exponential_decay(learning_rate=args.lr,
                                        global_step=global_step,
                                        decay_steps=args.decay_steps,
                                        decay_rate=args.decay_rate)
        tf.summary.scalar('lr', lr)

        # set optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # get specified gpus
        gpus = args.gpus.split(',')

        # get batch datas, [batch_size * gpu_num]
        batch_inputs, batch_labels, batch_pts_3d, batch_mask = \
            batch_data(args.data_path, args.batch_size, len(gpus))

        # get loss value
        tower_grads = []

        # get data for show
        outputs_all = []

        ### Tips: Multi gpus training set. ###
        # variable scope must set because of variable reuse
        with tf.variable_scope(tf.get_variable_scope()):
            # For each gpu device, call train
            for idx, _ in enumerate(gpus):
                # Because CUDA_VISIBLE_DEVICES given, so we redefine device to [0, len]
                with tf.device('/gpu:{}'.format(idx)):
                    # feed different datas
                    start = idx * args.batch_size
                    stop = (idx + 1) * args.batch_size
                    grad, outputs, loss = train(model, optimizer,
                                                batch_inputs[start:stop],
                                                batch_labels[start:stop],
                                                batch_mask[start:stop])

                    # Reuse variables for next devices.
                    tf.get_variable_scope().reuse_variables()
                    tower_grads.append(grad)

                    outputs_all.append(outputs)

                # showing for check
                if dev == '3':
                    tf.summary.image('output', outputs, max_outputs=5)
                    tf.summary.image('label', batch_labels, max_outputs=5)
        # same order of inputs
        outputs_all = tf.concat(outputs_all, axis=0)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # get train op
        train_op = optimizer.apply_gradients(grads, global_step)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses')
        loss = tf.reduce_mean(losses)
        tf.summary.scalar('loss', loss)

        # set saver
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            if not os.path.exists(args.output_path + '/checkpoint'):
                print('Begin init for model.')
                sess.run(init)
            else:
                print('Load ckpt init for model.')
                latest_ckpt = tf.train.latest_checkpoint(args.output_path)
                saver.restore(sess, latest_ckpt)

            # multi thread for data read
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # write log
            summary_writer = tf.summary.FileWriter(args.output_path, sess.graph)

            g_step = 0
            while g_step < args.max_steps:
                _, g_step = sess.run([train_op, global_step])
                if g_step % args.display_steps == 0:
                    # run and save summaries
                    cost, summary = sess.run([loss, summary_op])
                    summary_writer.add_summary(summary, g_step)

                    # show info
                    localtime = time.asctime(time.localtime(time.time()))
                    print("[{}] step:{} loss={:.5f}".format(localtime, g_step, cost))

                    # save model ckpt
                if g_step % (args.display_steps * 5) == 0:
                    saver.save(sess, args.output_path + '/model', g_step)

            # 停止所有线程
            coord.request_stop()
            coord.join(threads)


if __name__ == "__main__":
    args = parser.parse_args()
    # After we set this env param, the index of gpus redefine from 0 to len,
    # no matter what args.gpus specified.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)
