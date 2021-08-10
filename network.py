import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class Qnetwork(tf.keras.Model):

    def __init__(self, num_actions, learning_rate=0.001):

        super(Qnetwork, self()).__init__()

        self.num_actions = num_actions

        self.dense1 = layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.dense2 = layers.Dense(128, activation="relu", kernel_initializer="he_normal")
        self.out = layers.Dense(num_actions, kernel_initializer="he_normal") #action(one-hot)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    
    @tf.function
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.out(x)
        return out

    
    def predict(self, state):
        state = np.atleast_2d(state).astype(np.float32)
        return self(state).numpy()

    def update(self, states, selected_actions, target_values):
        selected_actions_onehot = tf.one_hot(selected_actions, self.num_actions)

        with tf.GradientTape as tape:
            #onehotを掛け選んだ行動のみを総和として算出(actionの次元をなくした)
            selected_actions_values = tf.reduce_sum(self(states) * selected_actions_onehot, axis=1) 

            #分散の２つ目の求め方
            loss = tf.reduce_mean(tf.square(target_values - selected_actions_values))

        variables = self.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))


