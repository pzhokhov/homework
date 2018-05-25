#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import os
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import pickle

def generate_expert_data(args):
    
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        returns, observations, actions = test_policy(
            policy_fn,
            envname=args.envname,
            num_rollouts=args.num_rollouts,
            render=args.render,
            max_timesteps=args.max_timesteps
        )
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions), 
                       'returns': returns}

        return expert_data


def test_policy(policy_fn, envname, num_rollouts, max_timesteps=None, render=False):
    import gym
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        
    return returns, observations, actions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    parser.add_argument('--expert-data', type=str, default=None)

    

    args = parser.parse_args()

    if args.expert_data and os.path.exists(args.expert_data):
        with open(args.expert_data, 'rb') as f:
            expert_data = pickle.load(f)
    else:    
        expert_data = generate_expert_data(args)

    if args.expert_data and not os.path.exists(args.expert_data):
        with open(args.expert_data, 'wb') as f:
            pickle.dump(expert_data, f)
    
    print('observations shape = {}'.format(expert_data['observations'].shape))
    print('actions shape = {}'.format(expert_data['actions'].shape))
    
    observations_shape = expert_data['observations'][0].shape
    num_actions = expert_data['actions'][0].shape[-1]
    from keras import Sequential
    from keras.layers import Dense

    model = Sequential()

    model.add(Dense(32, input_shape=observations_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))

    model.compile(optimizer='adam', loss='mse')
    model.fit(x=expert_data['observations'], y=expert_data['actions'].reshape(-1, num_actions), epochs=100, batch_size=128, validation_split=0.1)

    bc_policy = lambda o: model.predict(o)
        
    test_rollouts = 100
    returns, _, _ = test_policy(bc_policy, args.envname, test_rollouts, render=False)
       
    print('mean expert returns = {}'.format(np.mean(expert_data['returns'])))
    print('std expert returns = {}'.format(np.std(expert_data['returns'])))
    print('mean bc returns = {}'.format(np.mean(returns)))
    print('std bc returns = {}'.format(np.std(returns)))
    print('| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {} |'.format(
        args.envname, 
        np.mean(returns),
        np.std(returns),
        np.mean(expert_data['returns']),
        np.std(expert_data['returns']),
        args.num_rollouts
    ))
if __name__ == '__main__':
    main()
