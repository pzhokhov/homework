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



def load_expert(envname):
    return load_expert_file(os.path.join('experts', envname+'-v1.pkl'))

def load_expert_file(policy_file):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(policy_file)
    print('loaded and built')
    return policy_fn

def test_policy(policy_fn, envname, num_rollouts, max_timesteps=None, render=False):
    import gym
    env = gym.make(envname+'-v2')
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


def dagger(args):
    expert_policy  = load_expert(args.envname)
    expert_returns, expert_observations, expert_actions = test_policy(expert_policy, args.envname, args.num_rollouts)

    observations_shape = expert_observations[0].shape
    num_actions = expert_actions[0].shape[-1]

    mean_expert_return = np.mean(np.array(expert_returns))
    std_expert_return = np.std(np.array(expert_returns))

    current_model = None
    
    for i in range(args.dagger_iter):
        if args.dagger_reinitialize or current_model is None:
            current_model = build_model(observations_shape, num_actions)
            current_policy = lambda o: current_model.predict(o)            

        print('*** DAgger iteration {} ***'.format(i))

        current_model.fit(
            x=np.array(expert_observations),
            y=np.array(expert_actions).reshape(-1, num_actions),
            batch_size=args.dagger_batchsize,
            epochs=args.dagger_nepoch
        )
 
        returns, observations, actions = test_policy(current_policy, args.envname, args.num_rollouts)
        mean_return = np.mean(np.array(returns))
        std_return = np.std(np.array(returns))

        print('mean reward = {}'.format(np.mean(np.array(returns))))        
        print('std reward = {}'.format(np.std(np.array(returns))))

        actions = expert_policy(observations)

        expert_observations.append(observations)
        expert_actions.append(actions)
        
    return current_policy, mean_return, std_return, mean_expert_return, std_expert_return


def build_model(observations_shape, num_actions):
    from keras import Sequential
    from keras.layers import Dense

    model = Sequential()

    import pdb; pdb.set_trace()
    model.add(Dense(32, input_shape=observations_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='tanh'))

    model.compile(optimizer='adam', loss='mse')
 
    return model
        

def main():
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    parser.add_argument('--expert-data', type=str, default=None)

    parser.add_argument('--dagger-iter', type=int, default=1)
    parser.add_argument('--dagger-batchsize', type=int, default=128)
    parser.add_argument('--dagger-nepoch', type=int, default=1)
    parser.add_argument('--dagger-reinitialize', action='store_true', default=False)

    tf.Session().__enter__()
    tf_util.initialize()

    args = parser.parse_args()
    
    policy, mean_return, std_return, mean_expert_return, std_expert_return  = dagger(args)
         
    # test_rollouts = 100
    # returns, _, _ = test_policy(policy, args.envname, test_rollouts, render=False)
       
    print('| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {} |'.format(
        args.envname, 
        mean_return,
        std_return,
        mean_expert_return, 
        std_expert_return,
        args.num_rollouts
    ))
if __name__ == '__main__':
    main()
