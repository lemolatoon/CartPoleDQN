from typing import List, Tuple, Any
from six import int2byte
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym
import sys
import copy
from collections import deque
import random

#tmp
import pandas as pd
from tensorflow.python.autograph.operators.py_builtins import next_


class Agent:

    def __init__(self, num_actions, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber()):
        self.eps = 1.0
        self.num_actions = num_actions

        self.q_net = self.construct_model()
        self.q_net.compile(optimizer=optimizer, loss=loss)
        self.target_net = self.construct_model()
        self.target_net.compile(optimizer=optimizer, loss=loss)

        self.update_target_net() #deepcopyだめ

        self.t = 0

        self.FINAL_EPS = 0.1
        self.INITIAL_REPLAY_SIZE = 100
        self.NUM_REPLAY_MEMORY = 400000
        self.TRAIN_INTERVAL = 4
        self.TARGET_UPDATE_INTERVAL = 10000
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.epsilon_step = 1e-2
        

        self.episode_memory: deque = deque(maxlen=400000)
        

    def _get_action(self, state) -> int:
        q = sys.float_info.min
        n = 0
        for i in range(self.num_actions):
            #a = self.q_net.predict(state)
            x = np.concatenate([state, [[i]]], axis=1)
            pd.to_pickle(x, "x.pkl")
            a = self.q_net(x)
            #print(a)
            if q < a:
                n = i
        assert n >= 0 and n <= self.num_actions
        return n

    def get_action(self, state) -> int:
        if self.eps > self.FINAL_EPS and self.t >= self.INITIAL_REPLAY_SIZE:
            self.eps -= self.epsilon_step

        if self.eps < np.random.rand() or self.t < self.INITIAL_REPLAY_SIZE:
            return self._get_action(state)
        else:
            return np.random.randint(self.num_actions)


    def memory(self, state, action, reward, next_state, done):

        self.episode_memory.append((state, action, reward, next_state, done))

        if self.t > self.INITIAL_REPLAY_SIZE:
            if self.t % self.TRAIN_INTERVAL == 0:
                self.train_network()

            if self.t % self.TARGET_UPDATE_INTERVAL == 0:
                self.update_target_net()
        
        self.t += 1


    def train_network(self):
        print("学習するよ")
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        y_batch = []

        minibatch = random.sample(self.episode_memory, self.BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2]) 
            next_state_batch.append(data[3])
            done_batch.append(data[4])

        state_batch = np.array(state_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        next_state_batch = np.array(next_state_batch)
        done_batch = np.array(done_batch, dtype=int)

        target_q_values_batch = np.zeros((self.BATCH_SIZE, 1))
        for i in range(self.num_actions): #すべてのactionについてq値を計算
            #print(np.array(next_state_batch).shape)
            #print(np.concatenate([next_state_batch, np.full(shape=(self.BATCH_SIZE, 1), fill_value=i)], axis=1))
            x = np.concatenate([next_state_batch, np.full(shape=(self.BATCH_SIZE, 1), fill_value=i)], axis=1)
            y = self.target_net(np.array(x))
            print(f"y.shape{y.shape}")
            target_q_values_batch = np.maximum(y, target_q_values_batch)
            print(f"target_q_values_batch:{target_q_values_batch.shape}")
        y_batch = reward_batch + (1 - done_batch) * self.GAMMA * target_q_values_batch
        print(f"reward_batch:{reward_batch.shape}")
        print(f"1 - done_batch{(1 - done_batch).shape}")

        print(f"y_batch:{y_batch.shape}")
        x_train = np.concatenate([state_batch, action_batch], axis=1)
        print(f"x_train:{x_train.shape}")

        self.q_net.fit(np.concatenate([state_batch, action_batch], axis=1), y_batch, epochs=10)

    
    def update_target_net(self):
        for i in range(len(self.target_net.weights)):
            self.target_net.weights[i].assign(self.q_net.weights[i]) #代入


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



    def construct_model(self, input_shape=(5,)) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Dense(16, activation="relu")(inputs)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)


        #model = tf.keras.Sequential()
        #model.add(layers.Dense(16, activation="relu", input_shape=input_shape))
        #model.add(layers.Dense(3, input_shape=input_shape, activation=tf.keras.activations.linear))

        model.summary()

        return model




def main():
    NUM_EPISODES = 12000

    env: gym.Env = gym.make("CartPole-v0")
    agent = Agent(env.action_space.n)

    old_t = 0
    for i in tf.range(NUM_EPISODES):
        done = False
        state = env.reset()
        while not done:
            action = agent.get_action(state.reshape(1, 4))
            next_state, reward, done, _ = env.step(action)
            env.render()
            agent.memory(state, action, reward, next_state, done)
        if i % 10 == 0:
            print(f"Episode {i} finished after {agent.t - old_t} timesteps")
            old_t = agent.t
        

        


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