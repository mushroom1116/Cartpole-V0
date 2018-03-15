# -*- coding: utf-8 -*-
# @Time    : 3/14/18 8:05 PM
# @Author  : Wang Zhaorong
# @Site    : 
# @File    : ac_train.py
# @Software: PyCharm
from __future__ import print_function
from __future__ import division

import os
import argparse
import gym
import numpy as np
import tensorflow as tf

from brog_AC.brog_ac_agent import ActorCritic
from python_utils import *


def main(args):
    # INPUT_DIM = 80 * 80
    # HIDDEN_UNITS = 200
    # ACTION_DIM = 6
    INPUT_DIM = 4
    HIDDEN_UNITS = 100
    ACTION_DIM = 2
    MAX_EPISODES = 10000

    # load agent
    agent = ActorCritic(INPUT_DIM, HIDDEN_UNITS, ACTION_DIM)
    agent.construct_model(args.gpu)

    # load model or init a new
    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        # reuse saved model
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        mean_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        # build a new model
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        mean_rewards = None

    # load env
    env = gym.make('CartPole-v0')

    # training loop
    for ep in range(MAX_EPISODES):
        # reset env
        step = 0
        total_rewards = 0
        # state = preprocess(env.reset())
        state = env.reset()
        env.render()

        while True:
            # sample actions
            action = agent.sample_action(state[np.newaxis, :])
            # act!
            next_state, reward, done, _ = env.step(action)

            # next_state = preprocess(next_state)
            next_state = next_state

            step += 1
            total_rewards += reward

            agent.store_rollout(state, action, reward, next_state, done)
            # state shift
            state = next_state

            if done:
                break

        if mean_rewards is None:
            mean_rewards = total_rewards
        else:
            mean_rewards = 0.99 * mean_rewards + 0.01 * total_rewards

        rounds = (21 - np.abs(total_rewards)) + 21
        average_steps = (step + 1) / rounds
        print('Ep%s: %d rounds' % (ep_base + ep + 1, rounds))
        print('Average_steps: %.2f Reward: %s Average_reward: %.4f' %
              (average_steps, total_rewards, mean_rewards))

        # update model per episode
        agent.update_model()
        # model saving
        if ep > 0 and ep % args.save_every == 0:
            if not os.path.isdir(args.save_path):
                os.makedirs(args.save_path)
            save_name = str(round(mean_rewards, 2)) + '_' + str(ep_base + ep+1)
            saver.save(agent.sess, args.save_path + save_name)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='./model/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--save_every', default=100, help='Save model every x episodes')
    parser.add_argument(
        '--gpu', default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    return parser.parse_args()


def preprocess(obs):
    obs = obs[35:195]  # 160x160x3
    obs = obs[::2, ::2, 0]  # downsample (80x80)
    obs[obs == 144] = 0
    obs[obs == 109] = 0
    obs[obs != 0] = 1

    return obs.astype(np.float).ravel()

if __name__ == '__main__':
    main(args_parse())