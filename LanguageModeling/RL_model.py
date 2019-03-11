from __future__ import print_function
from __future__ import division


import tensorflow as tf
import numpy as np
from supercell_new import LSTMCell
from RL_brain import PolicyGradient


class adict(dict):
    ''' Attribute dictionary - a convenience data structure, similar to SimpleNamespace in python 3.3
        One can use attributes to read/write dictionary content.
    '''

    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self


def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]

    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError(
            "Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable(
            "Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def inference_graph(word_vocab_size=10000,  # configuration of medium
                    batch_size=20,
                    num_rnn_layers=2,
                    rnn_size=650,
                    num_unroll_steps=35,
                    n_actions=5,
                    dropout=0.0,
                    lamda=0.5
                    ):

    input_word = tf.placeholder(
        tf.int32, shape=[batch_size, num_unroll_steps], name="input")

    ''' First, embed characters '''
    with tf.variable_scope('Embedding'):
        embedding = tf.get_variable(
            "word_embedding", [word_vocab_size, rnn_size], dtype=tf.float32)
        input_embedded = tf.nn.embedding_lookup(embedding, input_word)
        if dropout != 0:
            input_embedded = tf.nn.dropout(input_embedded, 1. - dropout)

        ''' this op clears embedding vector of first symbol (symbol at position 0, which is by convention the position
        of the padding symbol). It can be used to mimic Torch7 embedding operator that keeps padding mapped to
        zero embedding vector and ignores gradient updates. For that do the following in TF:
        1. after parameter initialization, apply this op to zero out padding embedding vector
        2. after each gradient update, apply this op to keep padding at zero'''
        # clear_word_embedding_padding = tf.scatter_update(char_embedding, [0], tf.constant(0.0, shape=[1, char_embed_size]))

    ''' Finally, do LSTM '''
    with tf.variable_scope('LSTM'):
        RL = PolicyGradient(n_actions=n_actions, n_features=200)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(rnn_size)

        def attn_cell():
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=1. - dropout)
        cell1 = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(num_rnn_layers)])

        initial_rnn_state1 = cell1.zero_state(batch_size, dtype=tf.float32)

        inputs = tf.reshape(
            input_embedded, [batch_size, num_unroll_steps, rnn_size])
        inputs_list = [tf.squeeze(x, [1])
                       for x in tf.split(inputs, num_unroll_steps, 1)]

        layer1_outputs, final_rnn_state1 = tf.contrib.rnn.static_rnn(cell1, inputs_list,
                                                                     initial_state=initial_rnn_state1, dtype=tf.float32)

        cell2 = LSTMCell(rnn_size, RL, lamda)
        cell2 = tf.contrib.rnn.DropoutWrapper(
            cell2, output_keep_prob=1. - dropout)
        initial_rnn_state2 = cell2.zero_state(batch_size, dtype=tf.float32)
        layer2_outputs, final_rnn_state2 = tf.contrib.rnn.static_rnn(cell2, layer1_outputs,
                                                                     initial_state=initial_rnn_state2, dtype=tf.float32)
        # (time, batch, 1) => (batch, time)
        actions = tf.transpose(tf.squeeze(RL.actions))
        # (time, batch, n_actions) => (batch, time, n_actions)
        all_act_prob = tf.transpose(RL.all_act_prob, perm=(1, 0, 2))

        # linear projection onto output (word) vocab
        logits = []
        with tf.variable_scope('WordProjection') as scope:
            for idx, output in enumerate(layer2_outputs):
                if idx > 0:
                    scope.reuse_variables()
                logits.append(linear(output, word_vocab_size))

    return adict(
        input=input_word,
        # clear_char_embedding_padding = clear_char_embedding_padding,
        input_embedded=input_embedded,
        initial_rnn_state1=initial_rnn_state1,
        initial_rnn_state2=initial_rnn_state2,
        final_rnn_state1=final_rnn_state1,
        final_rnn_state2=final_rnn_state2,
        rnn_outputs=layer2_outputs,
        logits=logits,
        all_act_prob=all_act_prob,
        actions=actions
    )


def loss_graph(logits, all_act_prob, actions, word_vocab_size, batch_size, num_unroll_steps, n_actions):
    # loss function need to be the sum of reward!!!
    with tf.variable_scope('Loss'):
        targets = tf.placeholder(
            tf.int64, [batch_size, num_unroll_steps], name='targets')
        target_list = [tf.squeeze(x, [1])
                       for x in tf.split(targets, num_unroll_steps, 1)]
        loss_task = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=target_list), name='loss')

        # rewards = tf.transpose(tf.reduce_sum(tf.nn.softmax(logits) * tf.one_hot(target_list, word_vocab_size), axis = 2)) #reward shape(batch, time)
        # neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob+0.0000001)*tf.one_hot(actions, 5),axis = 2) #neg_log_prob shape(batch, time)
        # loss_RL = tf.reduce_mean(tf.reduce_sum(neg_log_prob * tf.expand_dims(tf.reduce_sum(rewards,1),-1),1))
        # loss_RL = tf.reduce_mean(tf.reduce_sum(neg_log_prob,1) * tf.reduce_sum(rewards, 1))
        rewards = tf.transpose(tf.reduce_sum(tf.nn.softmax(tf.stop_gradient(
            logits)) * tf.one_hot(tf.transpose(targets), word_vocab_size), axis=2))  # reward shape(batch, time)

        neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob+0.0000000001) * tf.one_hot(
            actions, n_actions), axis=2)  # neg_log_prob shape(batch, time)
        neg_log_prob_step = tf.matmul(neg_log_prob, tf.Variable(np.ones(
            num_unroll_steps, dtype=np.float32)-np.tri(num_unroll_steps, k=-1, dtype=np.float32), False))

        loss_RL = tf.reduce_mean(neg_log_prob_step * rewards) * 0.3

        loss_entropy = tf.reduce_mean(
            tf.log(all_act_prob+0.0000000001) * tf.one_hot(actions, n_actions))

    # loss_task = tf.Print(loss_task,[loss_task, loss_RL],message='loss_task and loss_RL')
    # loss_RL = tf.Print(loss_RL, [loss_RL], message='loss_RL')
    loss = loss_task + loss_RL / num_unroll_steps
    # loss = tf.Print(loss,[loss],message='loss')

    return adict(
        loss_task=loss_task,
        loss_entropy=loss_entropy,
        loss_RL=loss_RL,
        targets=targets,
        loss=loss
    )


def training_graph(loss, learning_rate=1.0, max_grad_norm=5.0):
    ''' Builds training graph. '''
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('SGD_Training'):
        # SGD learning parameter
        learning_rate = tf.Variable(
            learning_rate, trainable=False, name='learning_rate')

        # collect all trainable variables
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(
            tf.gradients(loss, tvars), max_grad_norm)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step)
        # optimizer = tf.train.AdamOptimizer(0.0001)
        # train_op = optimizer.minimize(loss)

    return adict(
        learning_rate=learning_rate,
        global_step=global_step,
        global_norm=global_norm,
        train_op=train_op)


def model_size():
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size


if __name__ == '__main__':

    with tf.Session() as sess:

        with tf.variable_scope('Model'):
            graph = inference_graph(
                char_vocab_size=51, word_vocab_size=10000, dropout=0.5)
            graph.update(loss_graph(
                graph.logits, batch_size=20, num_unroll_steps=35))
            graph.update(training_graph(
                graph.loss, learning_rate=1.0, max_grad_norm=5.0))

        with tf.variable_scope('Model', reuse=True):
            inference_graph = inference_graph(
                char_vocab_size=51, word_vocab_size=10000)
            inference_graph.update(loss_graph(
                graph.logits, batch_size=20, num_unroll_steps=35))

        print('Model size is:', model_size())

        # need a fake variable to write scalar summary
        tf.scalar_summary('fake', 0)
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter('./tmp', graph=sess.graph)
        writer.add_summary(sess.run(summary))
        writer.flush()
