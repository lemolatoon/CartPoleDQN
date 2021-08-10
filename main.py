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
import tqdm

import statistics

#tmp
import pandas as pd


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
        self.INITIAL_REPLAY_SIZE = 10000
        self.NUM_REPLAY_MEMORY = 40000
        self.TRAIN_INTERVAL = 4
        self.TARGET_UPDATE_INTERVAL = 10000
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.epsilon_step = 1e-4
        

        self.episode_memory: deque = deque(maxlen=self.NUM_REPLAY_MEMORY)
        

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
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        y_batch = []

        size = len(self.episode_memory)

        #minibatch = random.sample(self.episode_memory, self.BATCH_SIZE)
        minnibatch = self.episode_memory
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

        target_q_values_batch = np.zeros((self.BATCH_SIZE,)) #memory確保

        for i in range(self.num_actions): #すべてのactionについてq値を計算
            #print(np.array(next_state_batch).shape)
            #print(np.concatenate([next_state_batch, np.full(shape=(self.BATCH_SIZE, 1), fill_value=i)], axis=1))
            x = np.concatenate([next_state_batch, np.full(shape=(self.BATCH_SIZE, 1), fill_value=i)], axis=1) #(s, a)を結合
            y = self.target_net(np.array(x))
            y = y.numpy().reshape(-1) #縦のべくとるを横のベクトルにする
            target_q_values_batch = np.maximum(y, target_q_values_batch) #各アクションの中で最も大きいものを取る
        y_batch = reward_batch + (1 - done_batch) * self.GAMMA * target_q_values_batch #equationの右辺

        action_batch = action_batch.reshape(-1, 1) #state_batchと結合するためのreshape
        x_train = np.concatenate([state_batch, action_batch.reshape(-1, 1)], axis=1) #equation右辺のQの引数
        

        self.q_net.fit(np.concatenate([state_batch, action_batch], axis=1), y_batch, epochs=int(self.BATCH_SIZE * 4), verbose=0, batch_size=self.BATCH_SIZE) #verbose=0でno log
        #epochsが10だと少ない

    
    def update_target_net(self):
        for i in range(len(self.target_net.weights)):
            self.target_net.weights[i].assign(self.q_net.weights[i]) #代入
        
        x = np.ones(shape=(1, 5))
        assert self.q_net(x) == self.target_net(x) #計算結果が同じでなければならない




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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    agent = Agent(env.action_space.n, optimizer=optimizer)

    old_t = 0
    with tqdm.trange(NUM_EPISODES) as t:
        for i in t:
            done = False
            reward_sum = 0
            state = env.reset()
            rewards: deque = deque(maxlen=100)

            while not done:
                action = agent.get_action(state.reshape(1, 4))
                next_state, reward, done, _ = env.step(action)
                #print(f"reward:{reward}")
                reward_sum = reward_sum + reward
                #print(f"reward_sum:{reward_sum}")
                env.render()
                agent.memory(state, action, reward, next_state, done)

            rewards.append(reward_sum)
            avg_reward = statistics.mean(rewards)
        
            t.set_description(f"Episode {i}")
            t.set_postfix(running_reward=reward_sum, avg_reward=avg_reward, epsilon=agent.eps)

    pd.to_pickle("agent.pkl", agent)

        


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