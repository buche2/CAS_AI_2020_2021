import gym
import matplotlib.pyplot as plt
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from gym_minigrid.wrappers import ImgObsWrapper
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


class Agent:
    """Agent class for the cross-entropy learning algorithm.
    """

    def __init__(self, env):
        """Set up the environment, the neural network and member variables.

        Parameters
        ----------
        env : gym.Environment
            The game environment
        """
        self.env = env
        #self.observations = self.env.observation_space.shape[0]
        self.observations = 147
        self.actions = self.env.action_space.n
        self.model = self.get_model()

    def get_model(self):
        """Returns a keras NN model.
        """
        model = Sequential()
        model.add(Conv2D(32, (2, 2), activation='relu', input_shape=(56, 56, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(units=self.actions, activation='softmax'))
        model.compile(
            optimizer=Adam(lr=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def get_action(self, state):
        """Based on the state, get an action.
        """
        state = state.reshape(1, 56, 56, 3)  # 1 is the batch dim
        #state = state.reshape(1, -1) # [4,] => [1, 4]
        state = state/255.
        action = self.model(state, training=False).numpy()[0]
        action = np.random.choice(self.actions, p=action) # choice([0, 1], [0.5044534  0.49554658])
        return action

    def get_samples(self, num_episodes):
        """Sample games.
        """
        rewards = [0.0 for i in range(num_episodes)]
        episodes = [[] for i in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    def filter_episodes(self, rewards, episodes, percentile):
        """Helper function for the training.
        """
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observation = [step[0] for step in episode]
                action = [step[1] for step in episode]
                x_train.extend(observation)
                y_train.extend(action)
        x_train = np.asarray(x_train)
        y_train = to_categorical(y_train, num_classes=self.actions) # L = 0 => [1, 0]
        return x_train, y_train, reward_bound

    def train(self, percentile, num_iterations, num_episodes):
        """Play games and train the NN.
        """
        for iteration in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            x_train = x_train/255.
            self.model.fit(x=x_train, y=y_train, verbose=0)
            reward_mean = np.mean(rewards)
            print(f"Iteration - Reward mean: {reward_mean}, reward bound: {reward_bound}")
            if reward_mean > 500:
                break

    def play(self, num_episodes, render=True):
        """Test the trained agent.
        """
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print(f"Total reward: {total_reward} in episode {episode + 1}")
                    break


if __name__ == "__main__":
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = RGBImgPartialObsWrapper(env)  # Get pixel observations
    env = ImgObsWrapper(env)  # Get rid of the 'mission' field
    agent = Agent(env)
    print("Number of actions: ", agent.actions)
    agent.train(percentile=99.9, num_iterations=64, num_episodes=128)
    agent.play(num_episodes=3)
