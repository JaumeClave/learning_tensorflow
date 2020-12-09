# RNN intuition
import tensorflow as tf
from tensorflow import keras

my_rnn = RNN()
hidden_state = [0, 0, 0, 0]

sentence = ['I', 'love', 'recurrent', 'neural']

for word in sentence:
    prediction, hidden_state = my_rnn(word, hidden_state)

next_word_prediction = prediction
# 'networks'


# RNN from scratch
class MyRNNCell(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNNCell, self).__init__()

        # Initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim, rnn_units])

        #
        self_h = tf.zeros([rnn_units, 1])

    def call(self, x):
        # Update the hidden state
        self_h = tf.math.tanh(self.W_hh * self.h * self.W_xh * x)

        # Compute the output
        output = self.W_hy * self.h

