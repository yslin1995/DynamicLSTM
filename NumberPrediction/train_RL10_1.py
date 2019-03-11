import tensorflow as tf
import cPickle as pickle
import numpy as np
from supercell_new import LSTMCell
from RL_brain import PolicyGradient

batch_size = 100
time_step = 11
rnn_size = 200
learning_rate = 0.001
epoch = 200
n_actions = 10

train_set = np.array(pickle.load(open('num_train10_1.pkl', 'rb')))
test_set = np.array(pickle.load(open('num_test10_1.pkl', 'rb')))

inputs = tf.placeholder(dtype=tf.int32, shape=[batch_size, time_step])
inputs_ = tf.one_hot(inputs, 10)
labels = tf.placeholder(dtype=tf.int32, shape=[batch_size])

RL = PolicyGradient(
    n_actions=n_actions,
    n_features=10,
)

inputs_ = tf.unstack(inputs_, axis=1)  # input transfer to lstm form
lstm_cell = LSTMCell(rnn_size, RL)

outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, inputs=inputs_, dtype=tf.float32)
# (time-5, batch, 1) => (batch, time-5)
actions = tf.transpose(tf.squeeze(RL.actions))
# (time-5, batch, n_actions) => (batch, time-5, n_actions)
all_act_prob = tf.transpose(RL.all_act_prob, perm=(1, 0, 2))

output = outputs[-1]  # shape (batch_size, rnn_size)

layer = tf.layers.dense(
    inputs=output,
    units=10,
    activation=None,
    use_bias=True,
    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
    bias_initializer=tf.constant_initializer(0.1),
    name='fc3'
)

pred = tf.nn.softmax(layer)
loss_task = tf.reduce_mean(-tf.log(pred+0.0000001) * tf.one_hot(labels, depth=10))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(tf.one_hot(labels, depth=10), 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# reward = -1, 1
# shape(batch, time). first 5 actions XXXX
neg_log_prob = tf.reduce_sum(-tf.log(all_act_prob+0000001) * tf.one_hot(actions, n_actions), axis=2)
# rewards form 0,1 to -1,1 shape(batch)
rewards = tf.subtract(tf.cast(correct_pred, tf.float32), 0.5) * 2
rewards = tf.expand_dims(rewards, -1)
loss_RL = tf.reduce_mean(neg_log_prob * rewards)
loss_total = loss_task+loss_RL


train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
# train_task_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_task)
# train_RL_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_RL)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
model_path = "./model11.ckpt"
saver = tf.train.Saver()
best_acc = 0
count = 0

for j in range(epoch):
    i = 0
    print('****************  RL is beginning  *******************')
    while i * batch_size < 100000:
        input_x = train_set[i*batch_size:(i+1)*batch_size, :11]
        labels_y = train_set[i*batch_size:(i+1)*batch_size, 11]
        # print(input_x)
        # print(labels_y)
        acc, loss_, loss_RL_, actions_, _ = sess.run(
            fetches=[accuracy, loss_task, loss_RL, actions, train_op], feed_dict={inputs: input_x, labels: labels_y})
        if i % 200 == 0:
            print('the %d epoch the %d time accuracy is %f, loss is %f' %
                  (j, i, acc, loss_))
            print('loss_RL:', loss_RL_)
            print('last input', input_x[0])
            print('actions:', actions_[0])
        i += 1

    print('###############  TESTING  ####################')
    i = 0
    test_loss = []
    test_acc = []
    while i * batch_size < 1000:
        input_x = test_set[i*batch_size:(i + 1) * batch_size, :11]
        labels_y = test_set[i*batch_size:(i+1) * batch_size, 11]
        acc, loss_ = sess.run(fetches=[accuracy, loss_task], feed_dict={
                              inputs: input_x, labels: labels_y})
        test_loss.append(loss_)
        test_acc.append(acc)
        i += 1
    print('the TEST accuracy is %f, loss is %f' %
          (sum(test_acc)/len(test_acc), sum(test_loss)/len(test_loss)))
    cur_acc = sum(test_acc)/len(test_acc)
    if cur_acc > best_acc:
        best_acc = cur_acc
        saver.save(sess, model_path)
        print('===============================================>>>> SAVE MODEL')
        count = 0
    count += 1
    if count == 5:
        print('--------------------------------------------->>>>  EARLY STOP!!!')
