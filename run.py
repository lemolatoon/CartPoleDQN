import tensorflow as tf
import gym
from collections import deque
import numpy as np
import tqdm
import pandas as pd

import matplotlib.pyplot as plt

from network import Qnetwork

class Agent:

    BATCH_SIZE = 32

    GAMMA = 0.95
    
    def __init__(self, action_num):
        self.action_num = action_num
        
        self.q_net = Qnetwork(action_num)
        self.target_net = Qnetwork(action_num)

        self.weight_copy()

        self.epsilon = 0.1

    def weight_copy(self):
        self.target_net.set_weights(self.q_net.get_weights()) #重みをあわせる


    def get_action(self, state) -> int:
        n: int
        if self.epsilon < np.random.randn(): 
            n = np.argmax(self.q_net.predict(state))
        else:
            n = np.random.randint(self.action_num)
        assert n >= 0
        assert n < self.action_num
        return n

    def train_q_net(self, states, actions, rewards, next_states, dones):
        #right of equation (if done: value->r_{t+1})

        #s_{t+1}で得られるQ値のうち最大値(をとるaのときのQ値)
        next_Qs = np.max(self.target_net.predict(next_states), axis=1) 
        rights = [reward + self.GAMMA * next_Q if not done else reward for reward, next_Q, done in zip(rewards, next_Qs, dones)]

        self.q_net.update(np.array(states), np.array(actions), np.array(rights))


class GameMaster():

    ENV_NAME = "CartPole-v1"
    EXPERIENCE_SIZE = 10000

    WAIT_STEPS_BEFORE_TRAIN = 250
    EPISODES_NUM = 10000

    TRAIN_INTERVAL = 2
    WEIGHT_COPY_INTERVAL = 25

    RENDER: bool = False

    def __init__(self):
        self.env: gym.Env = gym.make(self.ENV_NAME)
        self.agent: Agent = Agent(action_num=self.env.action_space.n)

        self.episodes_list: deque = deque(maxlen=self.EXPERIENCE_SIZE)

        self.global_step = 0

    def one_step(self, state: np.ndarray, random: bool, render: bool=False):
        action: int
        if random:
            action = self.env.action_space.sample()
        else:
            action = self.agent.get_action(state)
        assert action >= 0 and action < self.env.action_space.n

        next_state, reward, done, _ = self.env.step(action)

        if render:
            self.env.render()

        self.episodes_list.append((state, action, reward, next_state, done)) #experience relay用の経験蓄積

        self.global_step += 1 #時間

        if not random:
            if self.global_step % self.TRAIN_INTERVAL == 0:
                states, actions, rewards, next_states, dones = self.get_minibatch()
                self.agent.train_q_net(states, actions, rewards, next_states, dones)
            if self.global_step % self.WEIGHT_COPY_INTERVAL == 0:
                self.agent.weight_copy()

        return (state, action, reward, next_state, done)

    def get_minibatch(self):
        #batch_sizeの数だけランダムにindex生成
        batch_mask = np.random.choice(len(self.episodes_list), size=self.agent.BATCH_SIZE, replace=False) 

        minibatch = [self.episodes_list[i] for i in batch_mask]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for data in minibatch:#state, action, reward, next_state, done
            states.append(data[0])
            actions.append(data[1])
            rewards.append(data[2])
            next_states.append(data[3])
            dones.append(data[4])
            
        return (states, actions, rewards, next_states, dones)

    def run_episode(self, random: bool) -> int:
        """run one episode
        return : sum of reward
        """
        state = self.env.reset()

        rewards = []

        done = False

        while not done:
            state, action, reward, next_state, done = self.one_step(state=state, random=random, render=self.RENDER)

            rewards.append(reward)

            state = next_state

        return np.sum(np.array(rewards))

    def run_episodes(self, episodes_num):
        rewards = []
        try:
            with tqdm.trange(episodes_num) as iter: #progress bar
                for episode in iter:
                    reward: int
                    if episode <= self.WAIT_STEPS_BEFORE_TRAIN:
                        #最低step数以下のときにはランダムアクションでepisode実行
                        reward = self.run_episode(random=True)
                        rewards.append(reward)
                    else:
                        reward = self.run_episode(random=False)
                        rewards.append(reward)

                    #直近100episodes平均
                    avg_reward = np.mean(np.array(rewards[-100:]))

                #progress bar description
                iter.set_description(f"Episode {episode}")
                iter.set_postfix(avg_reward=avg_reward ,reward=reward)

        except KeyboardInterrupt:
            print("train was interrupted as KeyboardInterrupt")
            self.history(rewards)

    def history(self, rewards):
        pd.to_pickle(rewards, "rewards.pkl")
        plt.plot(range(len(rewards)), rewards)
        plt.plot([0, 400], [195, 195], "--", color="darkred")
        plt.xlabel("episodes")
        plt.ylabel("Total Reward")
        plt.savefig("rewards_fig.jpg")

    def test(self):
        self.one_step(self.env.reset())

        
        


def main():
    master = GameMaster()
    master.run_episodes(master.EPISODES_NUM)


if __name__ == "__main__":
    main()