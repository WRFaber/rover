import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mars_microver_helpers import gradients_wrt_params, update_params
from mars_microver_policy import PolicyNet
from mars_microver_terrain import Terrain as MarsTerrain
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_episode(
    env: MarsTerrain, policy_net: PolicyNet, device="cpu", max_episode_len=100
):
    state = env.get_state_tensor(device)
    ep_length = 0
    done = False
    episode = []
    path = [tuple(env.rover_pos)]  # Start position
    while not done and ep_length < max_episode_len:
        ep_length += 1
        action_probs = policy_net(state).squeeze()
        probs_np = action_probs.detach().cpu().numpy()
        probs_np = probs_np / np.sum(probs_np)
        action = np.random.choice(np.arange(9), p=probs_np)
        log_prob = torch.log(action_probs[action])
        next_state_np, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(next_state_np).unsqueeze(0).to(device)
        episode.append((state, action, reward, log_prob))
        state = next_state
        path.append(tuple(env.rover_pos))  # Record new position
    return episode, path


def initialize_env(width=4.0, height=7.0, sandstorm_time=10):
    return MarsTerrain(width=width, height=height, sandstorm_time=sandstorm_time)


policy_net = PolicyNet()
policy_net.to(DEVICE)

episode_rewards = []

gamma = 0.99
lr_policy_net = 1e-5  # 2**-16
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr_policy_net)
model_path = "micro_policy_net.pth"
load = False
if os.path.exists(model_path) and load:
    policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"✅ Loaded model from {model_path}")
else:
    print("🚀 Starting with a fresh policy network")

width = 4.0
height = 7.0
sandstorm_time = 45
episode_lengths = []

for episode_num in tqdm(range(30000)):
    env = initialize_env(width=4.0, height=7.0, sandstorm_time=sandstorm_time)
    episode, path = generate_episode(
        env, policy_net=policy_net, device=DEVICE, max_episode_len=100
    )
    episode_lengths.append(len(episode))
    # Unpack episode
    rewards_ep = [reward for (_, _, reward, _) in episode]
    log_probs_ep = [log_prob for (_, _, _, log_prob) in episode]
    episode_rewards.append(np.sum(rewards_ep))
    # Policy gradient update
    returns = []
    for t in range(len(episode)):
        G = 0
        discount = 1
        for k in range(t, len(episode)):
            G += rewards_ep[k] * discount
            discount *= gamma
        returns.append(G)
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    for t in range(len(episode)):
        G_tensor = torch.tensor(
            returns[t], dtype=torch.float32, device=log_probs_ep[t].device
        )
        policy_loss = log_probs_ep[t] * G_tensor
        optimizer.zero_grad()
        gradients_wrt_params(policy_net, policy_loss)
        update_params(policy_net, lr_policy_net)

    # --- Plot every 1000 episodes ---
    if (episode_num + 1) % 2500 == 0:
        env.plot_path(episode_num)

# Save the trained model weights
torch.save(policy_net.state_dict(), "micro_policy_net.pth")
# dummy_input = torch.randn(1, 12)  # Adjust input size as needed
# torch.onnx.export(
#     policy_net,
#     dummy_input,
#     "micro_policy_net.onnx",
#     input_names=["input"],
#     output_names=["output"],
# )


# --- Plot at the end ---
env.plot_path(episode_num)

plt.figure()
plt.plot(range(len(episode_rewards)), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Episode Reward")
plt.title("Episode Reward Over Time")
plt.savefig("Micro_Time_Reward.png")

plt.figure()
window = 100  # Smoothing window
if len(episode_lengths) >= window:
    avg_lengths = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
else:
    avg_lengths = episode_lengths

plt.plot(avg_lengths)
plt.xlabel("Episode")
plt.ylabel("Average Time Steps per Episode")
plt.title("Average Time Spent per Episode")
plt.savefig("Micro_Time_SearchLength.png")
