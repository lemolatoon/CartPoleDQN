
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

import pandas as pd

def construct_model(input_shape=(5,)) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Dense(16, activation="relu")(inputs)
    outputs = layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

def sample1():
    q_net = construct_model()
    optimizer = tf.keras.optimizers.Adam()
    loss = tf.keras.losses.Huber()
    q_net.compile(optimizer=optimizer, loss=loss)
    q_net.summary()
    x = pd.read_pickle("x.pkl")
    print(x)
    y = q_net(x)
    print(y)

def sample():
    a = np.array([1, 2, 3, 4])
    print(a)
    a = a.reshape((4,))
    print(a)
    a = a.reshape((4, 1))
    print(a)


if __name__ == "__main__":
    sample()
