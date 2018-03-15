# -*- coding: utf-8 -*-
# @Time    : 3/13/18 10:03 AM
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
from AC.ac_agent import ActorCritic
import matplotlib.pyplot as plt


def main(args):
    env = gym.make('CartPole-v0')
    set_random_seed(seed=args.seed)
    env = env.unwrapped
    agent = ActorCritic(env, args)
    agent.construct_model(args.gpu)

    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        best_mean_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        best_mean_rewards = None

    rewards_history, steps_history = [], []
    train_steps = 0
    # Training
    for ep in range(args.max_ep):
        ep_rewards = 0
        state = env.reset()
        # env.render()

        for step in range(env.spec.timestep_limit):
            action = agent.sample_action(state[np.newaxis, :])
            next_state, reward, done, debug = env.step(action)
            train_steps += 1
            ep_rewards += reward
            reward = 0.1 if not done else -1
            agent.store_rollout(state,action,reward,next_state,done)
            state = next_state

            if done:
                break
        agent.update_model()
        steps_history.append(train_steps)
        if not rewards_history:
            rewards_history.append(ep_rewards)
        else:
            rewards_history.append(rewards_history[-1] * 0.9 + ep_rewards * 0.1)

        if ep % args.log_every == args.log_every - 1:
            total_rewards = 0
            for i in range(args.test_ep):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = agent.sample_action(state[np.newaxis, :])
                    state, reward, done, _ = env.step(action)
                    total_rewards += reward
                    if done:
                        break
            current_mean_rewards = total_rewards/args.test_ep
            print('Episode: %d Average Reward: %.2f' %(ep + 1, current_mean_rewards))
            print(agent.global_step)
            if best_mean_rewards is None or (current_mean_rewards >= best_mean_rewards):
                best_mean_rewards = current_mean_rewards
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_name = args.save_path + str(round(best_mean_rewards, 2)) + '_' + str(ep_base + ep + 1)
                saver.save(agent.sess, save_name)
                print('Model saved %s' % save_name)

    plt.plot(steps_history, rewards_history)
    plt.xlabel('steps')
    plt.ylabel('running avg rewards')
    plt.show()

def args_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model_path', default=None, help='whether to us a saved model. (*None|model path)')
    parse.add_argument('--save_path', default='./ac_model/', help='Path to save a model during training.')
    parse.add_argument('--gpu', default=-1, help='running on a specify gpu, -1 indicates using cpu.')
    parse.add_argument('--seed', default=31,help='random seed')
    parse.add_argument('--log_every', default=500, help='Log and save model every x episodes ')
    parse.add_argument('--test_ep',default=50)

    parse.add_argument('--hidden_units', default=100, help='hidden units in hidden layer')
    parse.add_argument('--max_ep', default=20000, help='Number of training episodes')
    parse.add_argument('--gamma', default=0.99, help='discounted factor')
    parse.add_argument('--lr', default=1e-4, help='Learning rate')
    parse.add_argument('--max_gradient', default=5, help='clipped gradient')
    parse.add_argument('--batch_size', default=300, help='size of training batch')
    return parse.parse_args()

def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

if __name__ == '__main__':
    main(args_parse())