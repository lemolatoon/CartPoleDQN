import tensorflow as tf
import numpy as np

class action_loss:

    def __init__(self, selected_actions, num_actions):
        self.selected_actions_onehot = tf.one_hot(selected_actions, num_actions) #選択したactionを保存

    

    def loss(self, states, target_values):
        """
        state : 状態
        target_values : ベルンマン方程式の右辺
        """
        #onehotを掛け選んだ行動のみを総和として算出(actionの次元をなくした)
        selected_actions_values = tf.reduce_sum(self(states) * self.selected_actions_onehot, axis=1) 

        #分散の２つ目の求め方
        loss = tf.reduce_mean(tf.square(target_values - selected_actions_values))

        return loss
