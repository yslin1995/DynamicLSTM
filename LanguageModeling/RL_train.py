# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf

import RL_model as model
import reader

flags = tf.flags

# data
flags.DEFINE_string('data_dir',     'data',     'data directory. Should contain train.txt/valid.txt/test.txt with input data')
flags.DEFINE_string('train_dir',    'cv/',      'training directory (models and summaries are saved there periodically)')
flags.DEFINE_string('load_model',   None,       '(optional) filename of the model to load. Useful for re-starting training from a checkpoint')

# model params
flags.DEFINE_integer('rnn_size',    650,    'size of LSTM internal state')
flags.DEFINE_integer('rnn_layers',  2,      'number of layers in the LSTM')
flags.DEFINE_float  ('dropout',     0.5,    'dropout. 0 = no dropout')
flags.DEFINE_float  ('lamda',       1,      'lambda value of RL')

# optimization
flags.DEFINE_float  ('learning_rate_decay', 1/1.2,  'learning rate decay')
flags.DEFINE_float  ('learning_rate',       1.0,    'starting learning rate')
flags.DEFINE_float  ('lr_threshold',        0.005,  'learning rate stop decreasing after threshold.')

flags.DEFINE_float  ('decay_when',          1.0,    'decay if validation perplexity does not improve by more than this much')
flags.DEFINE_float  ('param_init',          0.05,   'initialize parameters at')
flags.DEFINE_integer('num_unroll_steps',    35,     'number of timesteps to unroll for')
flags.DEFINE_integer('n_actions',           5,      'number of actions to take')
flags.DEFINE_integer('batch_size',          20,     'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          100,    'number of full passes through the training data')
flags.DEFINE_float  ('max_grad_norm',       5.0,    'normalize gradients at')

# bookkeeping
flags.DEFINE_integer('seed',            3435,   'random number generator seed')
flags.DEFINE_integer('print_every',     400,    'how often to print current loss')
flags.DEFINE_string ('EOS',             '+',    '<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')
flags.DEFINE_float  ('gpu_portion',     0.5,   'how much memory to used within a gpu')

FLAGS = flags.FLAGS

class PTBInput(object):
  """The input data."""
  def __init__(self, batch_size, num_steps, data, name=None):
    self.batch_size = batch_size
    self.num_steps = num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.data_queue = reader.ptb_producer(data, batch_size, num_steps, name=name)


def main(_):
    ''' Trains model from data '''

    # if not os.path.exists(FLAGS.train_dir): # directory for saving model
    #     os.mkdir(FLAGS.train_dir)
    #     print('-----Created training directory-----', FLAGS.train_dir)

    # LOAD DATA
    raw_data = reader.ptb_raw_data(FLAGS.data_dir)
    train_data, valid_data, test_data, _ = raw_data

    # each input below is like a stream, I guess so
    train_input = PTBInput(batch_size = FLAGS.batch_size, num_steps = FLAGS.num_unroll_steps,
                            data = train_data, name = "TrainInput")
    valid_input = PTBInput(batch_size = FLAGS.batch_size, num_steps = FLAGS.num_unroll_steps,
                            data = valid_data, name = "ValidInput")
    test_input  = PTBInput(batch_size = FLAGS.batch_size, num_steps = FLAGS.num_unroll_steps,
                            data = test_data, name = "TestInput")
    print('-----Initialized all dataset.-----')

    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_portion

    with tf.Session(config = tfConfig) as session:
        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed = FLAGS.seed)

        word_vocab_size = 10000 # fix word_vocab_size to 10000 as ptb_word_lm.py/MediumConfig

        ''' build training graph '''
        initializer = tf.random_uniform_initializer(-FLAGS.param_init, FLAGS.param_init)
        with tf.variable_scope("Model", initializer = initializer):
            print('-----Constructing training graph.-----')

            train_model = model.inference_graph(
                                                word_vocab_size = word_vocab_size,
                                                batch_size = FLAGS.batch_size,
                                                num_rnn_layers = FLAGS.rnn_layers,
                                                rnn_size = FLAGS.rnn_size,
                                                num_unroll_steps = FLAGS.num_unroll_steps,
                                                n_actions = FLAGS.n_actions,
                                                dropout = FLAGS.dropout,
                                                lamda = FLAGS.lamda
                                                )
            train_model.update(model.loss_graph(
                                    train_model.logits, train_model.all_act_prob, train_model.actions, word_vocab_size,
                                    FLAGS.batch_size, FLAGS.num_unroll_steps, FLAGS.n_actions))

            # scaling loss by FLAGS.num_unroll_steps effectively scales gradients by the same factor.
            # we need it to reproduce how the original Torch code optimizes. Without this, our gradients will be
            # much smaller (i.e. 35 times smaller) and to get system to learn we'd have to scale learning rate and max_grad_norm appropriately.
            # Thus, scaling gradients so that this trainer is exactly compatible with the original
            train_model.update(model.training_graph(train_model.loss * FLAGS.num_unroll_steps,
                                                    FLAGS.learning_rate, FLAGS.max_grad_norm))
            print('-----Constructed training graph.-----')

        # create saver before creating more graph nodes, so that we do not save any vars defined below
        saver = tf.train.Saver(max_to_keep = 50)

        ''' build graph for validation and testing (shares parameters with the training graph!) '''
        with tf.variable_scope("Model", reuse = True):
            print('-----Constructing validating graph.-----')
            valid_model = model.inference_graph(
                                                word_vocab_size = word_vocab_size,
                                                batch_size = FLAGS.batch_size,
                                                num_rnn_layers = FLAGS.rnn_layers,
                                                rnn_size = FLAGS.rnn_size,
                                                num_unroll_steps = FLAGS.num_unroll_steps,
                                                n_actions = FLAGS.n_actions,
                                                dropout = 0.0,
                                                lamda = FLAGS.lamda
                                                )
            valid_model.update(model.loss_graph(valid_model.logits, valid_model.all_act_prob, valid_model.actions, word_vocab_size,
                                                FLAGS.batch_size, FLAGS.num_unroll_steps, FLAGS.n_actions))
            print('-----Constructed validating graph.-----')

        with tf.variable_scope("Model", reuse = True):
            print('-----Constructing testing graph.-----')
            test_model = model.inference_graph(
                                                word_vocab_size = word_vocab_size,
                                                batch_size = FLAGS.batch_size,
                                                num_rnn_layers = FLAGS.rnn_layers,
                                                rnn_size = FLAGS.rnn_size,
                                                num_unroll_steps = FLAGS.num_unroll_steps,
                                                n_actions = FLAGS.n_actions,
                                                dropout = 0.0,
                                                lamda = FLAGS.lamda
                                              )
            test_model.update(model.loss_graph(test_model.logits, test_model.all_act_prob, test_model.actions, word_vocab_size,
                                                FLAGS.batch_size, FLAGS.num_unroll_steps, FLAGS.n_actions))
            print('-----Constructed testing graph.-----')

        if FLAGS.load_model:
            saver.restore(session, FLAGS.load_model)
            # print('Loaded model from', FLAGS.load_model, 'saved at global step', train_model.global_step.eval()）
            print ('---Restore model from', FLAGS.load_model)
            visual_action(session, train_model, train_input)
            print('---Visualization outputs End.')
            return

        else:
            tf.global_variables_initializer().run()
            # session.run(train_model.clear_char_embedding_padding)
            print('---Created and initialized fresh model. Size:', model.model_size())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph = session.graph)

        ''' take learning rate from CLI, not from saved graph '''
        session.run(tf.assign(train_model.learning_rate, FLAGS.learning_rate))

        ''' training starts here '''
        print('-----Start training model-----')
        best_valid_loss = None
        best_valid_pp = 99999.9
        best_valid_pp_epoch = 0
        best_test_pp = 99999.9

        rnn_state1 = session.run(train_model.initial_rnn_state1)
        rnn_state2 = session.run(train_model.initial_rnn_state2)

        for epoch in range(FLAGS.max_epochs): # max epochs for running code
            epoch_start_time = time.time()
            avg_train_loss = 0.0
            count = 0
            for stepInEpoch in range(train_input.epoch_size):
                x, y = session.run(train_input.data_queue)
                count += 1
                start_time = time.time()

                actions, loss_task, loss_RL, loss_entropy, total_loss, _, rnn_state1, rnn_state2, gradient_norm, step = session.run([
                    train_model.actions,
                    train_model.loss_task,
                    train_model.loss_RL,
                    train_model.loss_entropy,
                    train_model.loss,
                    train_model.train_op,
                    train_model.final_rnn_state1,
                    train_model.final_rnn_state2,
                    train_model.global_norm,
                    train_model.global_step
                    #train_model.clear_char_embedding_padding
                ], {
                    train_model.input  : x,
                    train_model.targets: y,
                    train_model.initial_rnn_state1: rnn_state1,
                    train_model.initial_rnn_state2: rnn_state2
                })
                # print('feed and fetch done.')
                avg_train_loss += 0.05 * (loss_task - avg_train_loss)

                time_elapsed = time.time() - start_time

                if count % FLAGS.print_every == 0:
                    print('%6d: %d [%5d/%5d], train_loss/perplexity = %6.8f/%6.7f secs/batch = %.4fs, grad.norm=%6.8f' % (step,
                                                            epoch, count,
                                                            train_input.epoch_size,
                                                            loss_task, np.exp(loss_task),
                                                            time_elapsed,
                                                            gradient_norm))
                    print('loss_RL:%f,loss_entropy:%f' % (loss_RL,loss_entropy))
                    print('actions', actions[0])

            print('Epoch training time:', time.time() - epoch_start_time)

            # epoch done: time to evaluate
            avg_valid_loss = 0.0
            count = 0
            rnn_state1 = session.run(valid_model.initial_rnn_state1)
            rnn_state2 = session.run(valid_model.initial_rnn_state2)

            for stepInEpoch in range(valid_input.epoch_size):
                x, y = session.run(valid_input.data_queue)
                count += 1
                start_time = time.time()

                loss, rnn_state1, rnn_state2 = session.run([
                    valid_model.loss_task,
                    valid_model.final_rnn_state1,
                    valid_model.final_rnn_state2
                ], {
                    valid_model.input  : x,
                    valid_model.targets: y,
                    valid_model.initial_rnn_state1: rnn_state1,
                    valid_model.initial_rnn_state2: rnn_state2
                })

                if count % FLAGS.print_every == 0:
                    print("\t> validation loss = %6.8f, perplexity = %6.8f" % (loss, np.exp(loss)))
                avg_valid_loss += loss / valid_input.epoch_size

            print("at the end of epoch:", epoch)
            print("train loss = %6.8f, perplexity = %6.8f" % (avg_train_loss, np.exp(avg_train_loss)))
            print("validation loss = %6.8f, perplexity = %6.8f" % (avg_valid_loss, np.exp(avg_valid_loss)))

            validation_pp = np.exp(avg_valid_loss)
            if validation_pp < best_valid_pp:
                best_valid_pp = validation_pp
                best_valid_pp_epoch = epoch
                #save_as = '%s/epoch%03d_%.4f.model' % (FLAGS.train_dir, epoch, best_valid_pp)
                #saver.save(session, save_as)
                #print('Saved model', save_as)

            rnn_state1 = session.run(test_model.initial_rnn_state1)
            rnn_state2 = session.run(test_model.initial_rnn_state2)
            count = 0
            avg_loss = 0
            start_time = time.time()
            for stepInEpoch in range(test_input.epoch_size):
                x, y = session.run(test_input.data_queue)
                count += 1
                loss, rnn_state1, rnn_state2 = session.run([
                    test_model.loss_task,
                    test_model.final_rnn_state1,
                    test_model.final_rnn_state2
                ], {
                    test_model.input: x,
                    test_model.targets: y,
                    test_model.initial_rnn_state1: rnn_state1,
                    test_model.initial_rnn_state2: rnn_state2
                })

                avg_loss += loss

            avg_loss /= count
            time_elapsed = time.time() - start_time

            print("test loss = %6.8f, perplexity = %6.8f" % (avg_loss, np.exp(avg_loss)))
            print("test samples:", count * FLAGS.batch_size, "time elapsed:", time_elapsed, "time per one batch:",
                  time_elapsed / count)

            if epoch == best_valid_pp_epoch:
                best_test_pp = np.exp(avg_loss)

            ''' write out summary events '''
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="train_loss", simple_value = avg_train_loss),
                tf.Summary.Value(tag="valid_loss", simple_value = avg_valid_loss)
            ])
            summary_writer.add_summary(summary, step)

            ''' decide if need to decay learning rate '''
            if best_valid_loss is not None and np.exp(avg_valid_loss) > np.exp(best_valid_loss) - FLAGS.decay_when:
                print('validation perplexity did not improve enough, decay learning rate')
                current_learning_rate = session.run(train_model.learning_rate)
                print('learning rate was:', current_learning_rate)
                current_learning_rate *= FLAGS.learning_rate_decay
                if current_learning_rate < FLAGS.lr_threshold:
                    current_learning_rate = FLAGS.lr_threshold
                    print('learning rate too small - stopping decay now')
                session.run(train_model.learning_rate.assign(current_learning_rate))
                print('new learning rate is:', current_learning_rate)
            else:
                best_valid_loss = avg_valid_loss

            print('----->[%d/%d]best_valid_pp:%.5f, best_test_pp:%.5f<-----'%(best_valid_pp_epoch, epoch, best_valid_pp, best_test_pp))

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()
