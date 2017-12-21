import tensorflow as tf
import numpy as np
import random
import config

class pgAgent():
    def __init__(self, env, nb_action, nb_warm_up, policy, testPolicy, gamma, lr, memory_limit, batchsize, train_interval):
        np.random.seed(1234567)
        tf.set_random_seed(1234567)

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

        self.episode_count = 0
        self.episode_reward = 0
        self.step_reward = np.zeros((len(config.Map.hole_pos)))
        self.gammas = [1.0]
        for i in range(len(config.Map.hole_pos)-1):
            self.gammas.append(self.gammas[-1]*self.gamma)
        self.gammas.reverse()

        self.true_ob_shape = []
        for d in self.ob_shape:
            self.true_ob_shape.append(d)

    def fit(self, nb_steps):
        observation = self.env.reset()
        for i in range(self.nb_warm_up):
            print "warm up step:", i
            action = self.get_action(observation)
            self.ob, self.r, done, info = self.env.step(action)
            self.save_memory(self.ob, action, self.r, done, (i+1) % self.train_interval)
            if done:
                observation = self.env.reset()

        for i in range(nb_steps):
            print "train step:", i
            action = self.get_action(observation)
            self.ob, self.r, done, info = self.env.step(action)
            self.save_memory(self.ob, action, self.r, done, (i+1) % self.train_interval)
            if (i+1) % self.train_interval == 0:
                batch_ob, batch_action, batch_reward = self.sample_memory(self.batch_size)
                self.sess.run(self.train_op, feed_dict={
                    self.tf_obs: np.array(batch_ob),  # shape=[None, n_obs]
                    self.tf_acts: np.array(batch_action),  # shape=[None, ]
                    self.tf_vt: batch_reward,  # shape=[None, ]
                })
                print "mean reward: ", np.mean(batch_reward)
            if done:
                observation = self.env.reset()

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
        self.policy.mask = self.env.get_mask()
        action = self.policy.select_action(probs[0])
        return action

    def get_test_action(self, observation):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        self.testPolicy.mask = self.env.get_mask()
        action = self.testPolicy.select_action(probs[0])
        return action

    def save_memory(self, state, action, reward, done, step):
        self.memory.append([state, action, reward, done, step])
        if done:
            hole_reward = self.env.hole_reward
            pt = len(self.memory) - 1
            count = 0
            self.episode_count += 1
            self.episode_reward += reward
            while pt >= 0 and count<len(config.Map.hole_pos):
                self.memory[pt-count][2] = hole_reward[self.memory[pt-count][1]] + reward * self.gammas[len(config.Map.hole_pos)-count-1]
                self.step_reward[len(config.Map.hole_pos)-1-count] += hole_reward[self.memory[pt-count][1]]
                count += 1

        if len(self.memory) > self.memory_limit:
            del self.memory[0]

    def sample_memory(self, batch_size):
        batch_ob = np.zeros([batch_size] + self.true_ob_shape)
        batch_action = np.zeros((batch_size, ))
        batch_reward = np.zeros((batch_size, ))
        # batch_reward = -np.ones((batch_size, ))
        for i in range(batch_size):
            index = random.randint(1, len(self.memory)-1)
            # index = -1
            step = self.memory[index][4]
            batch_ob[i] = self.memory[index][0]
            batch_action[i] = self.memory[index][1]
            # TODO more careful reward balance
            batch_reward[i] = self.memory[index][2] - self.step_reward[step]/self.episode_count - \
                              self.gammas[step] * self.episode_reward/self.episode_count

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


        reshape_ob = tf.reshape(self.tf_obs,[-1, 4, 5, len(config.Map.city_dis)])

        # conv1
        conv1 = tf.layers.conv2d(
            inputs=reshape_ob,
            filters=32,
            padding="SAME",
            kernel_size=[3, 3],
            activation=tf.sigmoid
        )

        # conv2
        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            padding="SAME",
            kernel_size=[3, 3],
            activation=tf.sigmoid
        )

        reshape_conv2 = tf.reshape(conv2, [-1, 4*5, 32])

        # fc1
        layer = tf.layers.dense(
            inputs=reshape_conv2,
            units=10,
            activation=tf.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        flat_layer = tf.reshape(layer, [-1, 4 * 5 * 10])
        # fc2
        all_act = tf.layers.dense(
            inputs=flat_layer,
            units=self.nb_action,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learningRate).minimize(loss)
