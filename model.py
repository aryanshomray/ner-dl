import tensorflow as tf
import numpy as np


class MyModel:

    def __init__(self,train, vec_size=300):

        self.train=train
        self.vec_size=vec_size
        self.model=None
        self.sent_length=20
        self.testing=None

    def model_compile(self):

        self.model=tf.keras.Sequential()

        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=150,
                                                                          activation='relu',
                                                                          input_shape=(self.train.shape[1],
                                                                                       self.train.shape[2]),

                                                                          return_sequences=True)))

        self.model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(9,activation='softmax')))

        self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    def training(self, train, labels, epoch=10, batch_size=64):

        self.model.fit(train,labels, epochs=epoch, batch_size=batch_size)
        print(self.model.summary())
    def save_model(self):

        self.model.save('my_model.h5')

    def load_model(self):

        self.model=tf.keras.models.load_model('mymodel.h5')


    def predict(self, predict):

        result=self.model.predict(predict)
        print(result)


