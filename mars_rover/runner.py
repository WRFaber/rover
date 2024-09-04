import matplotlib.pyplot as plt
import numpy as np
import torch
from helpers import gradients_wrt_params, update_params
from policy import PolicyNet
from terrain import Terrain
from tqdm import tqdm

ACTIONS = ["up", "down", "left", "right"]                   # Specify actions rover can take
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"     # Set device 

def generate_episode(grid: Terrain, policy_net: PolicyNet, device="cpu", max_episode_len = 100):
    """ Method to generate an episode for the rover in the specified terrain and following the specified policy

    Args:
        grid (Terrain): Represents the terrain.
        policy_net (PolicyNet): The policy network that maps state to action
        device (str, optional): Refers to the device in which content is to be shared and viewed, Defaults to "cpu".
        max_episode_len (int, optional): Madximum episode length. Defaults to 100.

    Yields:
        (state, action, reward), log_probs : a tuple providing current information about the environment and the generated log probability from the policy
    """
    state = grid.get_state(device)
    ep_length = 0
    while not grid.is_at_exit():
        # Convert state to tensor and pass through policy network to get action probabilities
        ep_length+=1
        action_probs = policy_net(state).squeeze()
        log_probs = torch.log(action_probs)
        nnorm = np.array(action_probs.detach().cpu().tolist())
        norm = nnorm/sum(nnorm)
        action = np.random.choice(np.arange(4), p=norm)

        # Take the action and get the new state and reward
        grid.move(ACTIONS[action])
        next_state = grid.get_state(device)
        reward = -0.1 if not grid.is_at_exit() else 0

        # Add the state, action, and reward to the episode
        new_episode_sample = (state, action, reward)
        yield new_episode_sample, log_probs

        # We do not want to add the state, action, and reward for reaching the exit position
        if reward == 0:
            break

        # Update the current state
        state = next_state
        if ep_length > max_episode_len:
            return

    # Add the final state, action, and reward for reaching the exit position
    new_episode_sample = (grid.get_state(device), None, 0)
    yield new_episode_sample, log_probs

def initialize_starting_place(n, m, exit_pos):
    x, y = exit_pos
    while (x,y) == exit_pos:
        x = np.random.choice(range(0,n))
        y = np.random.choice(range(0,m))
    return (x,y)


policy_net = PolicyNet()                                                       # The policy network
policy_net.to(DEVICE)                                                          # Mapped to device

lengths = []                                                                   # Pre-allocate lengths vector
rewards = []                                                                   # Pre-allocate rewards vector

gamma = 0.99                                                                   # Discount factor
lr_policy_net = 2**-16                                                         # Policy learning rate
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)        # Optimizer

prefix = "reinforce-per-step"                                                  # Psuedo name for model
n = 5                                                                          # Size of x-dim in grid for terrain
m = 5                                                                          # Size of y-dim in grid for terrain
exit_pos = (4,4)                                                               # Fixed exit position 


for episode_num in tqdm(range(1000)):
    all_iterations = []
    all_log_probs = []    
    initial_position = initialize_starting_place(n,m,exit_pos)
    terrain = Terrain(n,m,exit_pos,initial_position) 
    episode = list(generate_episode(terrain, policy_net=policy_net, device=DEVICE, max_episode_len=1000))
    lengths.append(len(episode))
    loss = 0
    for t, ((state, action, reward), log_probs) in enumerate(episode[:-1]):
        gammas_vec = gamma ** (torch.arange(t+1, len(episode))-t-1)
        # Since the reward is -1 for all steps except the last, we can just sum the gammas
        G = - torch.sum(gammas_vec)
        rewards.append(G.item())
        policy_loss = log_probs[action]
        optimizer.zero_grad()
        gradients_wrt_params(policy_net, policy_loss)
        update_params(policy_net, lr_policy_net  * G * gamma**t)

plt.plot(range(len(lengths)),lengths)
plt.show()
stop =0