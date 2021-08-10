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
from loss import action_loss
from network import Qnetwork


class Agent:

    def __init__(self, num_actions, optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber()):
        self.eps = 1.0
        self.num_actions = num_actions
        self.optimizer = optimizer
        
        self.FINAL_EPS = 0.1
        self.INITIAL_REPLAY_SIZE = 1000
        self.NUM_REPLAY_MEMORY = 20000
        self.TRAIN_INTERVAL = 4
        self.TARGET_UPDATE_INTERVAL = 250
        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.epsilon_step = 0.5 * 1e-4

        """
        self.q_net = self.construct_model(self.num_actions)
        self.q_net.compile(optimizer=optimizer)
        self.target_net = self.construct_model(self.num_actions)
        self.target_net.compile(optimizer=optimizer)
        """

        self.q_net = Qnetwork(self.num_actions)
        self.target_net = Qnetwork(self.num_actions)

        #self.q_net.set_weights(self.target_net.weights) #cant use deepcopy
        self.update_target_net()

        self.t = 0

        

        self.episode_memory: deque = deque(maxlen=self.NUM_REPLAY_MEMORY)
        

    def _get_action(self, state) -> int:
        q = sys.float_info.min
        n = np.argmax(self.q_net(state)) #Q値最大のactionを選択
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
                self.train_network_batch()

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
        minibatch = self.episode_memory
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

        #target_q_values_batch = np.zeros((self.BATCH_SIZE,)) #memory確保
        target_q_values_batch = np.zeros((size,)) #memory確保

        for i in range(self.num_actions): #すべてのactionについてq値を計算
            #print(np.array(next_state_batch).shape)
            #print(np.concatenate([next_state_batch, np.full(shape=(self.BATCH_SIZE, 1), fill_value=i)], axis=1))
            x = np.concatenate([next_state_batch, np.full(shape=(size, 1), fill_value=i)], axis=1) #(s, a)を結合
            y = self.target_net(np.array(x))
            y = y.numpy().reshape(-1) #縦のべくとるを横のベクトルにする
            target_q_values_batch = np.maximum(y, target_q_values_batch) #各アクションの中で最も大きいものを取る
        y_batch = reward_batch + (1 - done_batch) * self.GAMMA * target_q_values_batch #equationの右辺

        action_batch = action_batch.reshape(-1, 1) #state_batchと結合するためのreshape
        x_train = np.concatenate([state_batch, action_batch], axis=1) #equation右辺のQの引数
        

        self.q_net.fit(        #epochsが10だと少ない
            x_train, y_batch, epochs=5, verbose=0, batch_size=self.BATCH_SIZE) #verbose=0でno log


    def train_network_batch(self):
        states, actions, rewards, next_states, dones = self.get_minibatch()

        next_Qs = np.max(self.target_net.predict(next_states), axis=1) #max_a(Q_{target}(s', a))
        target_value = rewards * (1 - dones) * self.GAMMA * next_Qs 


        self.q_net.update(states=states, selected_actions=actions, target_values=target_value)

    def get_minibatch(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

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

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
    
    def update_target_net(self):
        for i in range(len(self.target_net.weights)):
            self.target_net.weights[i].assign(self.q_net.weights[i]) #代入
        print("==========================================")
        print(f"q_net{[w.numpy().shape for w in self.q_net.weights]}")
        print(f"target_net{[w.numpy().shape for w in self.target_net.weights]}")
        print("==========================================")
        #self.target_net.set_weights(self.q_net.weights)
        for i in range(len(self.target_net.weights)):
            self.target_net.weights[i].assign(self.q_net.weights[i]) #代入
        
        x = np.ones(shape=(self.BATCH_SIZE, 4))
        ys1 = self.q_net(x).numpy().reshape(-1)
        ys2 = self.target_net(x).numpy().reshape(-1)
        for y1, y2 in zip(ys1, ys2):
            assert y1 == y2 #計算結果が同じでなければならない




    def construct_model(self, num_actions, input_shape=(5,)) -> tf.keras.Model:
        """
        inputs = tf.keras.Input(shape=input_shape)
        x = layers.Dense(16, activation="relu")(inputs)
        outputs = layers.Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        """


        inputs = tf.keras.Input(shape=(4,)) #state
        x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(inputs)
        x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        outputs = layers.Dense(num_actions, kernel_initializer="he_normal")(x) #action(one-hot)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
                #env.render()
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