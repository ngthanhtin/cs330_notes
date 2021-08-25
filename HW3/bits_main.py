


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from BitFlip import BitFlipEnv
from buffers import Buffer
from utils import *
import tensorflow as tf
from model import Model

import argparse

parser = argparse.ArgumentParser(description='Flip Bits Environment with HER')

parser.add_argument('--HER', type=str, default='None',
                    help="None, Future, Final, Random")
parser.add_argument('--device', type=str, default='cuda',
                    help="use cuda")
parser.add_argument('--num-bits', type=int, default=7,
                    help="Length of bit vector")
parser.add_argument('--num-epochs',type=int, default=250,
                    help="Number of epochs to run training for")
parser.add_argument('--log-interval',type=int, default=5,
                    help='Epochs between printing log info')
parser.add_argument('--opt-steps',type=int, default=40,
                    help="Optimization steps in each epoch")

args = parser.parse_args()

# ************   Define global variables and initialize    ************ #

num_bits = args.num_bits  # number of bits in the bit_flipping environment
tau = 0.95  # Polyak averaging parameter
buffer_size = 1e6  # maximum number of elements in the replay buffer
batch_size = 128  # number of samples to draw from the replay_buffer

num_epochs = args.num_epochs  # epochs to run training for
num_episodes = 16  # episodes to run in the environment per epoch
num_relabeled = 4  # relabeled experiences to add to replay_buffer each pass
gamma = 0.98  # weighting past and future rewards


def update_replay_buffer(replay_buffer, episode_experience, HER):
    '''adds past experience to the replay buffer. Training is done with episodes from the replay
    buffer. When HER is used, relabeled experiences are also added to the replay buffer

    inputs: epsidode_experience - list of transitions from the last episode
    modifies: replay_buffer
    outputs: None'''

    for t in range(num_bits):
        # copy actual experience from episode_experience to replay_buffer

        # ======================== TODO modify code ========================
        s, a, r, s_, g = episode_experience[t]
        m = len(s) // 2
        # state
        inputs = s
        # next state
        inputs_ = s_
        # add to the replay buffer
        replay_buffer.add(inputs, a, r, inputs_)

        # when HER is used, each call to update_replay_buffer should add num_relabeled
        # relabeled points to the replay buffer

        if HER == 'None':
            # HER not being used, so do nothing
            pass

        elif HER == 'final':
            # final - relabel based on final state in episode
            # pass
            _, _, _, final_state, g_ = episode_experience[-1]
            new_goal = final_state[:m]
            # Update next_state as [next_state, new_goal_state]
            relabel_state = np.asarray([s_[:m], new_goal]).flatten()
            # Update reward (distance)
            r_new = -1 * np.sum(np.power((s_[:m] - new_goal), 2))
            replay_buffer.add(inputs, a, r_new, relabel_state)

        elif HER == 'future':
            # future - relabel based on future state. At each timestep t, relabel the
            # goal with a randomly select timestep between t and the end of the
            # episode
            # pass
            t_future = np.random.randint(t, m)
            _, _, _, relabel_goal, _ = episode_experience[t_future]
            replay_buffer.add(inputs, a, r, relabel_goal)


        elif HER == 'random':
            # random - relabel based on a random state in the episode
            #pass
            m = len(episode_experience)
            t_random = np.random.randint(0, m)
            _, _, _, relabel_goal, _ = episode_experience[t_random]
            replay_buffer.add(inputs, a, r, relabel_goal)

        # ========================      END TODO       ========================

        else:
            print("Invalid value for Her flag - HER not used")
    return

def update_target_model(model1, model2, tau):
    params1 = model1.named_parameters()
    params2 = model2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
            dict_params2[name1].data.copy_((1 - tau)*param1.data + tau*dict_params2[name1].data)

    model2.load_state_dict(dict_params2)

# ************   Main training loop    ************ #

def flip_bits(HER="None"):
    '''Main loop for running in the bit flipping environment. The DQN is
    trained over num_epochs. In each epoch, the agent runs in the environment
    num_episodes number of times. The Q-target and Q-policy networks are
    updated at the end of each epoch. Within one episode, Q-policy attempts
    to solve the environment and is limited to the same number as steps as the
    size of the environment

    inputs: HER - string specifying whether to use HER'''

    print("Running bit flip environment with %d bits and HER policy: %s" % (num_bits, HER))

    # create bit flipping environment and replay buffer
    bit_env = BitFlipEnv(num_bits)
    replay_buffer = Buffer(buffer_size, batch_size)

    # set up Q-policy (model) and Q-target (target_model)
    model = Model(num_bits).to(args.device)
    target_model = Model(num_bits).to(args.device)
    # Initialize Target model with policy model state dict
    # target_model.load_state_dict(model.state_dict())
    update_target_model(model, target_model, tau=0.0)
    target_model.eval()

    # define loss
    loss_func = nn.SmoothL1Loss(reduction='mean')
    # define optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    total_loss = []  # training loss for each epoch
    success_rate = []  # success rate for each epoch

    for i in range(num_epochs):
        # Run for a fixed number of epochs

        total_reward = 0.0  # total reward for the epoch
        successes = []  # record success rate for each episode of the epoch
        losses = []  # loss at the end of each epoch

        for k in range(num_episodes):
            # Run in the environment for num_episodes  

            state, goal_state = bit_env.reset()  # reset the environment
            # attempt to solve the environment
            # list for recording what happened in the episode
            episode_experience = []
            succeeded = False

            for t in range(num_bits):
                # attempt to solve the state - number of steps given to solve the
                # state is equal to the size of the vector

                # ======================== TODO modify code ========================
                #print(t, state, goal_state)
                inp_state = state
                # J: Concat goal_state to each state observation
                inp_state = np.concatenate((state, goal_state))
                inp_state = torch.FloatTensor(inp_state).to(args.device).unsqueeze(0)
                
                # forward pass to find action
                
                action = model(inp_state)
                action = np.argmax(action.detach().cpu().numpy(), axis=1)[0]
                
                # take the action
                next_state, reward, done, _ = bit_env.step(action)
                
                # J: In Goal cond. RL, reward=-distance(state, goal)
                r_func = 'l2'
                if r_func == 'sparse':
                    # -1 if not equal; 0 if equal
                    # -dirac{ state != goal_state }
                    reward = -1 * np.any(next_state != goal_state)
                elif r_func == 'l2':
                    # L2 norm (squared): ||different bits||^2
                    reward = -1 * np.sum(np.power( (next_state - goal_state), 2) )

                # J: Update state and next_state with goal_state (to sample from batch of experience later)
                state_g = np.asarray([state, goal_state]).flatten()
                next_state_g = np.asarray([next_state, goal_state]).flatten()
                # add to the episode experience (what happened)
                episode_experience.append((state_g, action, reward, next_state_g, goal_state))
                # calculate total reward
                total_reward += reward
                # update state
                state = next_state
                # mark that we've finished the episode and succeeded with training
                if done:
                    if succeeded:
                        continue
                    else:
                        succeeded = True

            total_reward = total_reward / args.log_interval
            successes.append(succeeded)  # track whether we succeeded in environment
            update_replay_buffer(replay_buffer, episode_experience, HER)  # add to the replay buffer; use specified  HER policy

        for k in range(args.opt_steps):
            # optimize the Q-policy network

            # sample from the replay buffer
            state, action, reward, next_state = replay_buffer.sample()
            state = torch.FloatTensor(state).to(args.device)
            next_state = torch.FloatTensor(next_state).to(args.device)
            action = torch.LongTensor(action).reshape(-1, 1).to(args.device)
            reward =  torch.FloatTensor(reward).reshape(-1, 1).to(args.device)
        
            curr_q_value = model(state).gather(1, action)

            # forward pass through target network
            next_q_value = target_model(next_state).max(dim=1, keepdim=True)[0].detach()
            
            # calculate target reward
            target_reward = torch.clamp(reward + gamma * next_q_value, -1./(1-gamma), 0)
            # calculate loss
            loss = loss_func(curr_q_value, target_reward)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            # append loss from this optimization step to the list of losses
            losses.append(loss.item())

            
        # target_model.load_state_dict(model.state_dict())  # update target model by copying Q-policy to Q-target
        update_target_model(model, target_model, tau)
        success_rate.append(np.mean(successes))  # append mean success rate for this epoch

        if i % args.log_interval == 0:
            print('Epoch: %d  Mean reward: %f  Success rate: %.4f Mean loss: %.4f' % (
                i, total_reward, np.mean(successes), np.mean(losses)))

    return success_rate


if __name__ == "__main__":
    # success_rate = flip_bits(HER=FLAGS.HER)  # run again with type of HER specified
    # # pass success rate for each run as first argument and labels as second list
    # plot_success_rate([success_rate], [FLAGS.HER])

    success_rate1 = flip_bits('None')  # run again with type of HER specified
    # pass success rate for each run as first argument and labels as second list
    

    success_rate2 = flip_bits(HER='future')
    success_rate3 = flip_bits(HER='final')
    success_rate4 = flip_bits(HER='random')


    plot_success_rate([success_rate1, success_rate2, success_rate3, success_rate4], ['None', 'Future', 'Final', 'Random'])