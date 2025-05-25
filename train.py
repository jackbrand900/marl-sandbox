import matplotlib.pyplot as plt
from agents.dqn_agent import DQNAgent
from envs.make_env import make_env
import numpy as np

env = make_env()

agents = {}
episode_rewards = {agent: [] for agent in env.agents}
losses = []

for agent in env.agents:
    obs_space = env.observation_space(agent)
    action_space = env.action_space(agent)
    
    print(f"Agent: {agent}")
    print(f"Observation space: {obs_space}")
    print(f"Observation space shape: {obs_space.shape}")
    print(f"Action space: {action_space}")
    print(f"Action space size: {action_space.n}")
    print("---")

    obs_size = int(np.prod(obs_space.shape))
    n_actions = action_space.n
    agents[agent] = DQNAgent(obs_size, n_actions)

n_episodes = 500
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.05

for ep in range(n_episodes):
    env.reset()
    done = False
    rewards_this_ep = {agent: 0 for agent in env.agents}
    total_loss = 0
    step_count = 0

    while not done:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if termination or truncation:
                action = None
            else:
                observation = np.array(observation).flatten()
                action = agents[agent].act(observation, epsilon)
            env.step(action)
            
            if action is not None:  # Only store and train if it's not a terminal state
                next_observation, next_reward, next_termination, next_truncation, next_info = env.last()
                next_observation = np.array(next_observation).flatten()
                agents[agent].store(observation, action, reward, next_observation, termination)
                loss = agents[agent].train_step()
                total_loss += loss
                rewards_this_ep[agent] += reward
                step_count += 1

            done = termination or truncation
            if done:
                break

    for agent in env.agents:
        episode_rewards[agent].append(rewards_this_ep[agent])
    losses.append(total_loss / step_count if step_count else 0)
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    if ep % 50 == 0:
        print(f"Episode {ep}: rewards = {rewards_this_ep}, avg_loss = {losses[-1]:.4f}")

# Plotting
for agent in episode_rewards:
    plt.plot(episode_rewards[agent], label=f"{agent} reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()

plt.plot(losses)
plt.xlabel("Episode")
plt.ylabel("Average Loss")
plt.title("Loss per Episode")
plt.show()