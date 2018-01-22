import tensorflow as tf
import numpy as np
import random
import config
import time
from tensorflow.contrib import rnn

class pgAgent():
    def __init__(self, env, nb_action, nb_warm_up, policy, testPolicy, gamma, lr, memory_limit, batchsize, train_interval):
        np.random.seed(1234567)
        tf.set_random_seed(123)

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

        self.test_reward = open("test_reward.txt",'w')

    def fit(self, nb_steps):
        observation = self.env.reset()
        for i in range(self.nb_warm_up):
            action = self.get_action(observation, i % config.Hole_num)
            self.ob, self.r, done, info = self.env.step(action)
            self.save_memory(self.ob, action, self.r, done, i % config.Hole_num)
            if done:
                print "warm up step:", i
                observation = self.env.reset()

        self.env.reset()
        time1 = time.time()
        epi_rewards = open('episodes.txt','w')

        for i in range(nb_steps):
            # print "train step:", i
            action = self.get_action(observation, i % config.Hole_num)
            self.ob, self.r, done, info = self.env.step(action)
            self.save_memory(self.ob, action, self.r, done, i % config.Hole_num)
            if (i+1) % self.train_interval == 0:
                print "train step:", i
                # print "before bp time:", (time.time() - time1) / 60
                batch_ob, batch_action, batch_reward, batch_steps = self.sample_memory(self.batch_size)
                self.sess.run(self.train_op, feed_dict={
                    self.tf_obs: np.array(batch_ob),  # shape=[None, n_obs]
                    self.tf_acts: np.array(batch_action),  # shape=[None, ]
                    self.tf_vt: batch_reward,  # shape=[None, ]
                    self.rnn_batch: self.batch_size,
                    self.current_step: batch_steps,
                })
                print "mean reward: ", np.mean(batch_reward)
            if done:
                # print self.r
                epi_rewards.write(str(self.r)+'\n')
                epi_rewards.write(str(self.env.new_city_hole.tolist())+'\n')
                print "round time:", (time.time() - time1)/60
                time1 = time.time()
                observation = self.env.reset()

    def test(self):
        observation = self.env.reset()
        step = 0
        while True:
            action = self.get_test_action(observation, step)
            self.ob, self.r, done, info = self.env.step(action)
            step += 1
            if done:
                # TODO print sth
                print self.r
                self.test_reward.write(str(self.r)+'\n')
                break


    def get_action(self, observation, step):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :],
                                                            self.rnn_batch: 1,
                                                            self.current_step: [step]})
        action = self.policy.select_action(probs[0])
        return action

    def get_test_action(self, observation, step):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :],
                                                            self.rnn_batch: 1,
                                                            self.current_step: [step]})
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
                # self.memory[pt-count][2] = hole_reward[self.memory[pt-count][1]] + reward * self.gammas[len(config.Map.hole_pos)-count-1]
                # self.step_reward[len(config.Map.hole_pos)-1-count] += hole_reward[self.memory[pt-count][1]]
                self.memory[pt - count][2] = reward
                count += 1

        if len(self.memory) > self.memory_limit:
            del self.memory[0]

    def sample_memory(self, batch_size):
        batch_ob = np.zeros([batch_size] + self.true_ob_shape)
        batch_action = np.zeros((batch_size, ))
        batch_reward = np.zeros((batch_size, ))
        batch_steps = np.zeros((batch_size, ), dtype=np.int32)
        # batch_reward = -np.ones((batch_size, ))
        for i in range(batch_size):
            index = random.randint(1, len(self.memory)-1)
            # index = -1
            step = self.memory[index][4]
            batch_ob[i] = self.memory[index][0]
            batch_action[i] = self.memory[index][1]
            # TODO more careful reward balance
            batch_reward[i] = self.memory[index][2] - self.episode_reward/self.episode_count
                              # - self.step_reward[step]/self.episode_count - \
                              # self.gammas[step] * self.episode_reward/self.episode_count
            batch_steps[i] = step
        return batch_ob, batch_action, batch_reward, batch_steps

    def get_net(self):
        # TODO a more carefully designed net
        with tf.name_scope('inputs'):
            true_ob_shape = [None]
            for d in self.ob_shape:
                true_ob_shape.append(d)
            self.tf_obs = tf.placeholder(tf.float32, true_ob_shape, name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
            self.rnn_batch = tf.placeholder(tf.int32, [])
            self.current_step = tf.placeholder(tf.int32, [None,])


        cell = rnn.BasicLSTMCell(len(config.Map.city_dis), forget_bias=1.0,
                                            state_is_tuple=True,activation=tf.nn.tanh)

        initial_state = cell.zero_state(self.rnn_batch, dtype=tf.float32)

        lstm_out, final_state = tf.nn.dynamic_rnn(cell, self.tf_obs, initial_state=initial_state,
                                                  sequence_length=self.current_step, time_major=False)

        final_h = final_state[-1]

        # fc1
        layer = tf.layers.dense(
            inputs=final_h,
            units=100,
            activation=tf.sigmoid,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        # flat_layer = tf.reshape(layer, [-1, 210 * 10])
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,
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
