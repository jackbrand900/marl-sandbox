from pettingzoo.mpe import simple_spread_v2
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecEnvWrapper
import supersuit as ss
import numpy as np
import imageio
from PIL import Image

# ✅ Wrapper to remove `truncated` key for SB3 compatibility
class TruncateWrapper(VecEnvWrapper):
    def step_wait(self):
        obs, reward, done, truncated, info = self.venv.step_wait()
        return obs, reward, done, info
    def reset(self):
        return self.venv.reset()

# ✅ 1. Load and pad training environment
env = simple_spread_v2.parallel_env()
env = ss.pad_observations_v0(env)
env = ss.pad_action_space_v0(env)
env.reset()

# ✅ 2. Confirm consistent observation shapes
shapes = [env.observation_spaces[a].shape for a in env.possible_agents]
print("Observation shapes:", shapes)
assert all(s == shapes[0] for s in shapes), "❌ Inconsistent observation shapes"

# ✅ 3. Create vectorized environment for SB3
vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class='stable_baselines3')
vec_env = TruncateWrapper(vec_env)

# ✅ 4. Train each agent
models = {}
for agent in env.possible_agents:
    print(f"Training {agent}...")
    model = DQN("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=5000)
    models[agent] = model
    model.save(f"dqn_{agent}")

print("✅ Training complete. Recording test episodes...")

# ✅ 5. Setup test environment for video recording
test_env = simple_spread_v2.parallel_env(render_mode="rgb_array")
test_env = ss.pad_observations_v0(test_env)
test_env = ss.pad_action_space_v0(test_env)

frames = []
NUM_EPISODES = 5

for ep in range(NUM_EPISODES):
    obs = test_env.reset()
    done = {agent: False for agent in test_env.agents}
    step_count = 0

    while not all(done.values()):
        actions = {
            agent: models[agent].predict(obs[agent], deterministic=True)[0]
            for agent in test_env.agents if not done[agent]
        }
        obs, reward, done, truncated, info = test_env.step(actions)

        frame = test_env.render()
        if frame is not None:
            # ✅ Resize frame to 704x704 to ensure compatibility
            resized_frame = np.array(Image.fromarray(frame).resize((704, 704)))
            frames.append(resized_frame)

        step_count += 1

    print(f"✅ Episode {ep + 1} completed after {step_count} steps")

test_env.close()

# ✅ Save debug frame to check visibility
Image.fromarray(frames[-1]).save("debug_frame.png")
print("📸 Saved last frame as debug_frame.png")

# ✅ Save the video
imageio.mimsave("test_run.mp4", frames, fps=15)
print("🎥 Saved video as test_run.mp4")
