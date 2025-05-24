from pettingzoo.mpe import simple_tag_v2
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnvWrapper
import supersuit as ss
import numpy as np

# ✅ Wrapper to remove `truncated` key for SB3 compatibility
class TruncateWrapper(VecEnvWrapper):
    def step_wait(self):
        obs, reward, done, truncated, info = self.venv.step_wait()
        return obs, reward, done, info

    def reset(self):
        return self.venv.reset()

# ✅ 1. Load and pad training environment BEFORE reset
env = simple_tag_v2.parallel_env()
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
env.reset()

# ✅ 2. Confirm observation shapes are consistent
shapes = [env.observation_spaces[agent].shape for agent in env.possible_agents]
print("Observation shapes:", shapes)
assert all(s == shapes[0] for s in shapes), "❌ Inconsistent observation shapes!"

# ✅ 3. Convert to vectorized SB3-compatible env
vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class='stable_baselines3')
vec_env = TruncateWrapper(vec_env)

# ✅ 4. Train separate DQN for each agent
models = {}
for agent in env.possible_agents:
    print(f"Training {agent}...")
    model = DQN("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=5000)
    models[agent] = model
    model.save(f"dqn_{agent}")

print("✅ Training complete. Running test rollout...")

# ✅ 5. Create and pad test environment
test_env = simple_tag_v2.parallel_env(render_mode="human")
test_env = ss.pad_observations_v0(test_env)
test_env = ss.pad_action_space_v0(test_env)
obs = test_env.reset()
done = {agent: False for agent in test_env.agents}

# ✅ 6. Run test rollout
while not all(done.values()):
    actions = {
        agent: models[agent].predict(obs[agent], deterministic=True)[0]
        for agent in test_env.agents if not done[agent]
    }
    obs, reward, done, truncated, info = test_env.step(actions)
    test_env.render()

test_env.close()
