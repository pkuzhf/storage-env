import tensorflow as tf
import numpy as np
import random
import config

class pgAgent():
    def __init__(self, env, nb_action, nb_warm_up, policy, testPolicy, gamma, lr, memory_limit, batchsize, train_interval):
        np.random.seed(1)
        tf.set_random_seed(1)

        self.env = env
        self.nb_action = nb_action
        self.ob_shape = env.reset().shape
        self.policy = policy
        self.testPolicy = testPolicy
        self.gamma = gamma
        self.learningRate = lr
        self.memory = []
        self.memory_limit = memory_limit
        self.nb_warm_up = nb_warm_up
        self.batch_size = batchsize
        self.train_interval = train_interval

        self.get_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.ob = None
        self.r = None

        self.true_ob_shape = []
        for d in self.ob_shape:
            self.true_ob_shape.append(d)

    def fit(self, nb_steps):
        observation = self.env.reset()
        for i in range(self.nb_warm_up):
            action = self.get_action(observation)
            self.ob, self.r, done, info = self.env.step(action)
            self.save_memory(self.ob, action, self.r, done)
            if done:
                observation = self.env.reset()

        for i in range(nb_steps):
            print "train step:", i
            action = self.get_action(observation)
            self.ob, self.r, done, info = self.env.step(action)
            self.save_memory(self.ob, action, self.r, done)
            if i % self.train_interval == 0:
                batch_ob, batch_action, batch_reward = self.sample_memory(self.batch_size)
                self.sess.run(self.train_op, feed_dict={
                    self.tf_obs: np.array(batch_ob),  # shape=[None, n_obs]
                    self.tf_acts: np.array(batch_action),  # shape=[None, ]
                    self.tf_vt: batch_reward,  # shape=[None, ]
                })
                print "mean reward: ", np.mean(batch_reward)

    def test(self):
        observation = self.env.reset()
        while True:
            action = self.get_test_action(observation)
            self.ob, self.r, done, info = self.env.step(action)
            if done:
                # TODO print sth
                print self.r

    def get_action(self, observation):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = self.policy.select_action(probs[0])
        return action

    def get_test_action(self, observation):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = self.testPolicy.select_action(probs[0])
        return action

    def save_memory(self, state, action, reward, done):
        self.memory.append([state, action, reward, done])
        if len(self.memory) > self.memory_limit:
            del self.memory[0]

    def sample_memory(self, batch_size):
        batch_ob = np.zeros([batch_size] + self.true_ob_shape)
        batch_action = np.zeros((batch_size, ))
        batch_reward = np.zeros((batch_size, ))
        for i in range(batch_size):
            index = random.randint(0,len(self.memory)-10)
            batch_ob[i] = self.memory[index][0]
            batch_action[i] = self.memory[index][1]
            for j in range(9, -1, -1):
                # discount reward
                if self.memory[index+j][3]:
                    batch_reward[i] = 0
                batch_reward[i] = batch_reward[i] * self.gamma + self.memory[index+j][2]
        print batch_action
        print batch_reward
        return batch_ob, batch_action, batch_reward

    def get_net(self):
        # TODO a more carefully designed net
        with tf.name_scope('inputs'):
            true_ob_shape = [None]
            for d in self.ob_shape:
                true_ob_shape.append(d)
            self.tf_obs = tf.placeholder(tf.float32, true_ob_shape, name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")

        trans_ob = tf.transpose(self.tf_obs, [0,3,1,2])
        reshape_ob = tf.reshape(trans_ob,[-1,config.Map.Width,config.Map.Width,1])

        # conv1
        conv1 = tf.layers.conv2d(
            inputs=reshape_ob,
            filters=32,
            padding="SAME",
            kernel_size=[3, 3],
            activation=tf.nn.relu
        )

        # conv2
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            padding="SAME",
            kernel_size=[3, 3],
            activation=tf.nn.relu
        )

        reshape_conv2 = tf.reshape(conv2, [-1, 8, config.Map.Width, config.Map.Width, 32])

        # fc1
        layer = tf.layers.dense(
            inputs=reshape_conv2,
            units=10,
            activation=tf.nn.relu,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        flat_layer = tf.reshape(layer, [-1, 8 * config.Map.Width*config.Map.Height * 10])
        # fc2
        all_act = tf.layers.dense(
            inputs=flat_layer,
            units=self.nb_action,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learningRate).minimize(loss)
