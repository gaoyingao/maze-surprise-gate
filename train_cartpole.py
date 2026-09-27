from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")
model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
            batch_size=64, gamma=0.99, verbose=1, seed=42)
model.learn(total_timesteps=100_000)
model.save("ppo_cartpole")

env_eval = gym.make("CartPole-v1")
rewards = []
for ep in range(20):
    obs, _ = env_eval.reset(seed=1000 + ep)
    done, total = False, 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, _ = env_eval.step(action)
        total += r
        done = terminated or truncated
    rewards.append(total)
print("平均得分:", sum(rewards) / len(rewards))