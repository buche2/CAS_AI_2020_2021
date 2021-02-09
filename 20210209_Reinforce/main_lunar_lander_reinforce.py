import collections
import os
import torch

import gym
import matplotlib.pyplot as plt
import numpy as np
from reinforce_torch import PolicyGradientAgent
from typing import Deque, List

PROJECT_PATH = os.path.abspath("/Users/cedricmoullet/sandbox/CAS_AI_2020_2021/20210209_Reinforce")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_PATH = os.path.join(MODELS_PATH, "reinforce_lunarlander.pt")
PLOTS_PATH = os.path.join(PROJECT_PATH, "plots")

# box2d has to be installed: pip3 install box2d

def plot_learning_curve(scores, agent: PolicyGradientAgent, n_games: int):
    fname = 'REINFORCE_' + 'lunar_lunar_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = PLOTS_PATH + '/' + fname + '.png'
    x = [i + 1 for i in range(len(scores))]

    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


def train(env, agent: PolicyGradientAgent, n_games: int):
    best_reward_mean = -200
    scores = []
    for i in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            score += reward
            agent.store_rewards(reward)
            observation = observation_

        agent.learn()
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score)

        if avg_score > best_reward_mean:
            best_reward_mean = avg_score
            torch.save(agent.policy.state_dict(), MODEL_PATH)
            print(f"New best mean: {best_reward_mean}")

    return scores

def play(env, agent, n_games: int):
    scores = []

    for i in range(n_games):
        env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            score += reward
            env.render()
        print('episode ', i, 'score %.1f' % score)
        scores.append(score)
    return scores

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 3000
    agent = PolicyGradientAgent(gamma=0.99, lr=0.0005, input_dims=[8],
                                n_actions=4)

    if os.path.exists(MODEL_PATH):
        agent.policy.load_state_dict(torch.load(MODEL_PATH))

    # Train the model
    scores = train(env, agent, n_games)
    plot_learning_curve(scores, agent, n_games)

    # Play with the trained model
    # input("Play?")
    scores = play(env, agent, n_games=20)
    print(f"Scores mean: {np.mean(scores)}")
