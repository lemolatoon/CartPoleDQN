from typing import List, Tuple, Any
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym
import sys
import copy
from collections import deque

class Agent:

    def __init__(self, num_actions, optimizer=tf.keras.optimizers.Adam()):
        self.eps = 0.7
        self.INITIAL_REPLAY_SIZE = 200
        self.num_actions = num_actions

        self.q_net = self.construct_model()
        #self.q_net.compile(optimizer=optimizer)
        self.target_net = copy.deepcopy(self.q_net)

        self.episode_memory: deque = deque()

    def _get_action(self, state) -> int:
        q = sys.float_info.min
        n = 0
        for i in range(self.num_actions):
            #a = self.q_net.predict(state)
            a = self.q_net(state)
            #print(a)
            if q < a:
                n = i
        assert n >= 0 and n <= self.num_actions
        return n

    def get_action(self, state) -> int:
        if self.eps < np.random.rand() or self.t < self.INITIAL_REPLAY_SIZE:
            return self._get_action(state)
        else:
            return np.random.randint(self.num_actions)


    def run_episode(self, initial_state: np.array, model: tf.keras.Model, max_steps: int, eps=0.3) -> Any:
        """
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        next_states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        """

        episode = np.zeros((max_steps, 5))
        

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            state = tf.expand_dims(state, 0)



    def construct_model(self, input_shape=(4,)) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Dense(16, activation="relu")(inputs)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        #model.summary()

        return model


class Experience():

    def __init__(self):
        pass


def main():
    NUM_EPISODES = 12000

    env: gym.Env = gym.make("CartPole-v0")
    agent = Agent(env.action_space.n)

    for i in tf.range(NUM_EPISODES):
        done = False
        state = env.reset()
        while not done:
            action = agent.get_action(state.reshape(1, 4))
            next_state, reward, done, _ = env.step(action)
            env.render()
        print(f"Episode {i} finished")
        

        


if __name__ == "__main__":
    main()


def sample():
    env = gym.make("CartPole-v0")
    for i in range(10):
        obserbation = env.reset()
        for t in range(1000):
            env.render()
            steps = env.step(env.action_space.sample())
            print(steps)
            observation, reward, done, info = steps            
            if done:
                print(f"Episode {i} finished after {t} timesteps")
                break

    env.close()