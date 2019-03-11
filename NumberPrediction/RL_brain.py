import numpy as np
import tensorflow as tf

# reproducible

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.actions = []
        self.all_act_prob = []

    def choose_action(self, observation, cur_time):
        observation = tf.stop_gradient(observation)
        layer = tf.layers.dense(
            inputs=observation,
            units=self.n_features,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        ) # shape(batch*time, n_actions)

        act_prob = tf.nn.softmax(all_act, name='act_prob')

        actions = tf.multinomial(tf.log(act_prob), 1)

        if cur_time!=0:
            self.actions.append(actions)
            self.all_act_prob.append(act_prob)

        return actions

