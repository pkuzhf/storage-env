import tensorflow as tf
import numpy as np
import random
import config
import time
from tensorflow.contrib import rnn
import copy

class pgAgent():
    def __init__(self, env, nb_action, nb_warm_up, policy, testPolicy, gamma, lr, memory_limit, batchsize,
                 train_interval, pre_train_epis):
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
        self.saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state("models/")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.ob = None
        self.r = None

        self.episode_count = 0
        self.episode_reward = 0
        self.pre_train_epis = pre_train_epis
        self.step_reward = np.zeros((len(config.Map.hole_pos)))
        self.gammas = [1.0]
        for i in range(len(config.Map.hole_pos)-1):
            self.gammas.append(self.gammas[-1]*self.gamma)
        self.gammas.reverse()

        self.true_ob_shape = []
        for d in self.ob_shape:
            self.true_ob_shape.append(d)

        self.test_reward = open("test_reward.txt",'w')

        self.best_postfix = np.zeros((config.Hole_num))

        self.nb_city = len(config.Map.city_dis)
        self.nb_hole = len(config.Map.hole_pos)
        self.evaluate_interval = config.evaluate_interval
        self.nb_exchange = config.nb_exchange

    def fit(self, nb_steps):
        observation = self.env.reset()
        for i in range(self.nb_warm_up):
            print "warm up step:", i
            while True:
                action = self.get_action(observation)
                self.ob, self.r, done, info = self.env.step(action)
                observation = self.ob
                self.save_memory(observation, action, self.r, done, info)
                if done:
                    observation = self.env.reset()
                    break

        observation = self.env.reset()
        time1 = time.time()
        # epi_rewards = open('episodes.txt','w')

        for i in range(nb_steps):
            print "train step:", i
            while True:
                # print "train step:", i
                action = self.get_action(observation)
                self.ob, self.r, done, info = self.env.step(action)
                observation = self.ob
                self.save_memory(observation, action, self.r, done, info)
                if done:
                    # print "before bp time:", (time.time() - time1) / 60
                    batch_ob, batch_action, batch_reward= self.sample_memory(self.batch_size)
                    self.sess.run(self.train_op, feed_dict={
                        self.tf_obs: np.array(batch_ob),  # shape=[None, n_obs]
                        self.tf_acts: np.array(batch_action),  # shape=[None, ]
                        self.tf_vt: batch_reward,  # shape=[None, ]
                    })
                    # time.sleep(1)
                    # print "mean reward: ", np.mean(batch_reward)
                if done:
                    # print self.r
                    # epi_rewards.write(str(self.r)+'\n')
                    # epi_rewards.write(str(self.env.new_city_hole.tolist())+'\n')
                    # print "round time:", (time.time() - time1)/60
                    time1 = time.time()
                    observation = self.env.reset()
                    break

    def test(self):
        for i in range(5):
            observation = self.env.reset()
            step = 0
            while True:
                action = self.get_test_action(observation)
                self.ob, self.r, done, info = self.env.step(action)
                observation = self.ob
                step += 1
                if done:
                    # TODO print sth
                    print self.r
                    print info
                    self.test_reward.write(str(self.r)+'\n')
                    self.test_reward.write(str(info) + '\n')
                    self.test_reward.close()
                    self.test_reward = open("test_reward.txt",'a')
                    break
        self.saver.save(self.sess, "models/model.ckpt")

    def get_action(self, observation):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        self.policy.set_mask(self.env.mask)
        action = self.policy.select_action(probs[0])
        return action

    def get_test_action(self, observation):
        probs = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        self.testPolicy.set_mask(self.env.mask)
        action = self.testPolicy.select_action(probs[0])
        return action

    def save_memory(self, state, action, reward, done, v):
        self.memory.append([state, action, reward, done, v]) # pre_sum and v
        if done:
            self.pre_train_epis -= 1
            current_v = v
            pt = len(self.memory) - 2
            self.episode_count += 1
            self.episode_reward += reward - v
            while not self.memory[pt][3]:
                self.memory[pt][2] = reward
                if self.memory[pt][4]>0:
                    current_v = self.memory[pt][4]
                self.memory[pt][4] = current_v
                self.episode_count += 1
                self.episode_reward += self.memory[pt][2] - self.memory[pt][4]
                pt-=1

        if len(self.memory) > self.memory_limit:
            self.episode_count -= 1
            self.episode_reward -= self.memory[0][2] - self.memory[0][4]
            del self.memory[0]


    def sample_memory(self, batch_size):
        batch_ob = np.zeros([batch_size] + self.true_ob_shape)
        batch_action = np.zeros((batch_size, ))
        batch_reward = np.zeros((batch_size, ))
        # batch_reward = -np.ones((batch_size, ))
        for i in range(batch_size):
            index = random.randint(1, len(self.memory)-2)
            # index = -1
            batch_ob[i] = self.memory[index][0]
            batch_action[i] = self.memory[index][1]
            batch_reward[i] = self.memory[index][2] - self.episode_reward/self.episode_count - self.memory[index][4]
            #
            # batch_ob[i] = self.memory[0][0]
            # batch_action[i] = 1
            # batch_reward[i] = 10
        print batch_reward

        return batch_ob, batch_action, batch_reward

    def get_net(self):
        # TODO a more carefully designed net
        with tf.name_scope('inputs'):
            true_ob_shape = [None]
            for d in self.ob_shape:
                true_ob_shape.append(d)
            self.tf_obs = tf.placeholder(tf.int32, true_ob_shape, name="observations")
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")


        stacked = tf.stack([self.tf_obs],axis=3)

        cnn_ob = tf.reshape(stacked,[-1,5,4,1])

        cnn_ob = tf.cast(cnn_ob, tf.float32)

        conv1 = tf.layers.conv2d(
            inputs=cnn_ob,
            filters=16,
            padding="SAME",
            kernel_size=[3, 3],
            activation=tf.sigmoid
        )

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=16,
            padding="SAME",
            kernel_size=[3, 3],
            activation=tf.sigmoid
        )

        flat_ob0 = tf.reshape(conv2,[-1, 2, 20*16])

        fc0 = tf.layers.dense(
            inputs=flat_ob0,
            units=64,
            activation=tf.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc0'
        )

        flat_ob1 = tf.reshape(fc0, [-1, 128])

        # fc1
        fc1 = tf.layers.dense(
            inputs=flat_ob1,
            units=32,
            activation=tf.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )

        # fc1 = tf.Print(fc1, [fc1], message="fc1: ", summarize=32)

        # fc2
        all_act = tf.layers.dense(
            inputs=fc1,
            units=self.nb_action,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # all_act = tf.Print(all_act,[all_act],message="last_act: ", summarize=20)
        # around one

        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability


        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            # reg_loss = tf.reduce_sum(tf.square(all_act))
            reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in tf.trainable_variables()])
            pg_loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            loss = pg_loss + 0.01 * reg_loss


        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.learningRate).minimize(loss)
