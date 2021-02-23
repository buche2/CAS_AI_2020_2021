import collections
import os
import random
from typing import Deque

import gym
import numpy as np
import matplotlib.pyplot as plt

from cartpoleA2CNN import Actor
from cartpoleA2CNN import Critic

PROJECT_PATH = os.path.abspath("/Users/cedricmoullet/sandbox/CAS_AI_2020_2021/20210216_PolicyGradient_A2C_V2")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
ACTOR_PATH = os.path.join(MODELS_PATH, "actor_cartpole.h5")
CRITIC_PATH = os.path.join(MODELS_PATH, "critic_cartpole.h5")


class Agent:
    def __init__(self, env: gym.Env):
        self.env = env
        self.num_observations = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n
        self.num_values = 1
        self.gamma = 0.95
        self.learning_rate_actor = 1e-3 # 0.001
        self.learning_rate_critic = 5e-3 # 0.005
        self.actor = Actor(
            self.num_observations,
            self.num_actions,
            self.learning_rate_actor
        )
        self.critic = Critic(
            self.num_observations,
            self.num_values,
            self.learning_rate_critic
        )
        self.batch_size = 16
        self.memory_size = 16

        self.memory = collections.deque(maxlen=self.memory_size)

    def get_action(self, state: np.ndarray):
        policy = self.actor(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action

    def update_policy(self):
        # make a batch see https://github.com/cedricmoullet/CAS_AI_2020_2021/blob/main/20210202_DQN/cartPoleDqnAgent.py
        values = np.zeros(shape=(self.batch_size, self.num_values)) # (1, 1)
        advantages = np.zeros(shape=(self.batch_size, self.num_actions)) # (1, 2)

        states, actions, rewards, states_next, dones = zip(*self.memory)

        value = self.critic(np.concatenate(states))
        next_value = self.critic(np.concatenate(states_next))

        for i in range(self.batch_size):
            action = actions[i]
            done = dones[i]
            if done:
                advantages[i][action] = rewards[i] - value[i]
                values[i][0] = rewards[i]
            else:
                advantages[i][action] = (rewards[i] + self.gamma * next_value[i]) - value[i]
                values[i][0] = rewards[i] + self.gamma * next_value[i] - value[i]

        self.actor.fit(np.array(states).reshape(self.memory_size, 4), advantages)
        self.critic.fit(np.array(states).reshape(self.memory_size, 4), values)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, num_episodes: int):
        last_rewards: Deque = collections.deque(maxlen=5)
        all_rewards = []

        for episode in range(1, num_episodes + 1):
            self.memory.clear()
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            mem = 0

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                if done and total_reward < 499:
                    reward = -100.0
                self.remember(state, action, reward, next_state, done)
                mem += 1
                if mem == self.memory_size:
                    self.update_policy()
                    mem = 0
                    self.memory.clear()

                total_reward += reward
                state = next_state

                if done:
                    if total_reward < 500:
                        total_reward += 100.0
                    last_rewards.append(total_reward)
                    current_reward_mean = np.mean(last_rewards)
                    all_rewards.append(total_reward)
                    print(f"Episode: {episode} Reward: {total_reward} MeanReward: {current_reward_mean}")

                    if current_reward_mean > 400:
                        self.actor.save_model(ACTOR_PATH)
                        self.critic.save_model(CRITIC_PATH)
                        return
                    break
        self.actor.save_model(ACTOR_PATH)
        self.critic.save_model(CRITIC_PATH)

        plt.plot(all_rewards)
        plt.show()

    def play(self, num_episodes: int, render: bool = True):
        self.actor.load_model(ACTOR_PATH)
        self.critic.load_model(CRITIC_PATH)

        episode_rewards = []

        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, newshape=(1, -1)).astype(np.float32)

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                total_reward += reward
                state = next_state

                if done:
                    print(f"Episode: {episode} Reward: {total_reward}")
                    episode_rewards.append(total_reward)
                    break

        print(f"Mean reward from {num_episodes} episodes = {np.mean(episode_rewards)}")
        print(f"Best reward = {np.max(episode_rewards)}")

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=1_000)
    input("Press <Enter> to play?")
    agent.play(num_episodes=10, render=True)
