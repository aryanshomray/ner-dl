import tensorflow as tf
from tensorflow.keras.layers import LSTM
import numpy as np

class MyModel:
    def __init__(self):
        self.model=LSTM