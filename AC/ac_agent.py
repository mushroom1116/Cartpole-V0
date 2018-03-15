# -*- coding: utf-8 -*-
# @Time    : 3/13/18 10:03 AM
# @Author  : Wang Zhaorong
# @Site    :
# @File    : ac_agent.py
# @Software: PyCharm

from __future__ import print_function
from __future__ import division
import numpy as np
import tensorflow as tf

class ActorCritic(object):
    def __init__(self, env, args):
        self.state_dim = env.observation_space.shape[0]  # 4
        self.action_dim = env.action_space.n  # 2
        self.hidden_units = args.hidden_units  # 10
        self.gamma = args.gamma  # 0.99
        self.lr = args.lr  # 1e-4
        self.max_gradient = args.max_gradient  # 5
        self.batch_size = args.batch_size  # 128

        self.ep_count = 0
        self.global_step = 0
        self.buffer_reset()

    def actor_network(self, input_state):
        # 初始化用的是random_normal/sqrt(n_in)
        w1 = tf.Variable(tf.div(tf.random_normal([self.state_dim, self.hidden_units]), np.sqrt(self.state_dim)), name='w1')
        b1 = tf.Variable(tf.constant(0.0, shape=[self.hidden_units]),name='b1')
        # 第一层激活函数用relu
        h1 = tf.nn.relu(tf.matmul(input_state, w1) + b1)
        w2 = tf.Variable(tf.div(tf.random_normal([self.hidden_units, self.action_dim]), np.sqrt(self.hidden_units)),name='w2')
        b2 = tf.Variable(tf.constant(0.0, shape=[self.action_dim]), name='b2')
        # actor网络输出与动作维度一致
        # logp = tf.nn.softmax(tf.matmul(h1, w2) + b2)
        logp = tf.matmul(h1, w2) + b2

        return logp

    def critic_network(self,input_state):
        w1 = tf.Variable(tf.div(tf.random_normal([self.state_dim, self.hidden_units]),np.sqrt(self.state_dim)),name='w1')
        b1 = tf.Variable(tf.constant(0.0,shape=[self.hidden_units]),name='b1')
        h1 = tf.nn.relu(tf.matmul(input_state,w1)+b1)
        w2 = tf.Variable(tf.div(tf.random_normal([self.hidden_units,1]),np.sqrt(self.hidden_units)),name='w2')
        b2 = tf.Variable(tf.constant(0.0, shape=[1]), name='b2')
        # critic网络输出为s的值，维度为1
        state_value = tf.matmul(h1,w2) + b2

        return state_value

    def construct_model(self, gpu):
        if gpu == -1:
            device = '/cpu:0'
            sess_config = tf.ConfigProto()
        else:
            device = '/gpu:' + str(gpu)
            sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)

        with tf.device(device):
            with tf.name_scope('model_inputs'):
                self.input_state = tf.placeholder(tf.float32, [None, self.state_dim])
            with tf.name_scope('actor_network'):
                self.logp = self.actor_network(self.input_state)
            with tf.name_scope('critic_network'):
                self.state_value = self.critic_network(self.input_state)

            # get parameters
            actor_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
            critic_parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_network')

            self.taken_action = tf.placeholder(tf.int32, [None,])
            self.discounted_rewards = tf.placeholder(tf.float32, [None, 1])

            # optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            # actor loss
            # tf.nn.sparse_softmax_cross_entropy_with_logits 其中logits是[None, num_class],
            # labels是[None]其中每个元素是标签索引（互斥类）
            # 先对logits做softmax，将其在一维上每个元素映射为概率，然后对概率做log运算，得到每个概率的log值，
            # 然后根据labels给的索引（因为是互斥类，类标签相当于三one-hot，只需要一个索引即可）在log概率中找到对应位置的元素,再加上负号即可
            # 返回值与labels的shape一致，与logits类型一致
            self.actor_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logp, labels=self.taken_action)
            # self.actor_loss = tf.reduce_sum(-tf.log(self.logp)*tf.one_hot(self.taken_action,self.action_dim), axis=1)
            # advantage
            self.advantage = (self.discounted_rewards - self.state_value)[:, 0]
            # actor gradient
            actor_gradients = tf.gradients(self.actor_loss, actor_parameters, self.advantage)
            # zip打包为元组的列表,每个元素是(actor_gradients, actor_parameters)
            self.actor_gradients = list(zip(actor_gradients, actor_parameters))

            # critic loss
            self.critic_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.state_value))
            # critic gradient
            # 返回(gradients, parameter)对的list
            self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_parameters)
            # 将两个list相加，相当于list1.extend(list2)
            self.gradients = self.actor_gradients + self.critic_gradients

            # clip gradient
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (tf.clip_by_value(
                        grad, -self.max_gradient, self.max_gradient), var)
            with tf.name_scope('train_actor_critic'):
                # train operation
                self.train_op = self.optimizer.apply_gradients(self.gradients)

    def sample_action(self, state):
        self.global_step += 1
        def softmax(x):
            max_x = np.amax(x)
            e = np.exp(x-max_x)
            # e = np.exp(x)
            return e / np.sum(e)

        logp = self.sess.run(self.logp, {self.input_state: state})[0]
        prob = softmax(logp) - 1e-5
        # 对actor的输出softmax之后,根据白努力分布采样，根据该概率掷1次筛子，取最大概率(出现1)的索引
        action = np.argmax(np.random.multinomial(1, prob))
        # action = np.argmax(logp)
        return action

    def update_model(self):
        state_buffer = np.array(self.state_buffer)
        action_buffer = np.array(self.action_buffer)
        # 按照行顺序（垂直）将数组对叠起来
        discounted_rewards_buffer = np.vstack(self.reward_discount())
        ep_steps = len(action_buffer)
        # 返回一个数组，内容与range的相同
        shuffle_index = np.arange(ep_steps)
        # 乱序重拍
        # np.random.shuffle(shuffle_index)

        # for i in range(0, ep_steps, self.batch_size):
        #     if self.batch_size <= ep_steps:
        #         end_index = i + self.batch_size
        #     else:
        #         end_index = ep_steps
        #     batch_index = shuffle_index[i:end_index]
        #     # get batch from buffer
        #     input_state = state_buffer[batch_index]
        #     taken_action = action_buffer[batch_index]
        #     discounted_rewards = discounted_rewards_buffer[batch_index]
        #
        #     # train
        #     self.sess.run(self.train_op, feed_dict={self.input_state:input_state,
        #                                             self.taken_action:taken_action,
        #                                             self.discounted_rewards:discounted_rewards})



        input_state = state_buffer[shuffle_index]
        taken_action = action_buffer[shuffle_index]
        discounted_rewards = discounted_rewards_buffer[shuffle_index]

        # train
        self.sess.run(self.train_op, feed_dict={self.input_state: input_state,
                                                self.taken_action: taken_action,
                                                self.discounted_rewards: discounted_rewards})
        # cleanup
        self.buffer_reset()

        self.ep_count += 1

    def store_rollout(self, state, action, reward, next_state, done):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_state_buffer.append(next_state)
        self.done_buffer.append(done)

    def buffer_reset(self):
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_state_buffer = []
        self.done_buffer = []

    def reward_discount(self):
        r = self.reward_buffer
        d_r = np.zeros_like(r)
        running_add = 0
        # 从后往前遍历list
        for t in range(len(r))[::-1]:
            # if r[t] != 0:
            #     running_add = 0
            # game boundary. reset the running add
            running_add = r[t] + running_add * self.gamma
            d_r[t] += running_add
        # standardize the rewards
        d_r -= np.mean(d_r)
        d_r /= np.std(d_r)
        return d_r




