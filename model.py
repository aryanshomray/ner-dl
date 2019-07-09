
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from tensorflow.keras.layers import Bidirectional,TimeDistributed,Dense,LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model

tensorboard=TensorBoard(log_dir='logs/{}'.format(time()))


class MyModel:

    def __init__(self,train, vec_size=300):

        self.train=train
        self.vec_size=vec_size
        self.model=None
        self.sent_length=20
        self.testing=None

    def model_compile(self):

        self.model=Sequential()

        self.model.add(Bidirectional(LSTM(units=150,
                                                                          activation='relu',
                                                                          input_shape=(self.train.shape[1],
                                                                                       self.train.shape[2]),

                                                                          return_sequences=True)))

        self.model.add(TimeDistributed(Dense(9,activation='softmax')))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def training(self, train, labels, epoch=10, batch_size=64):

        self.model.fit(train,labels, epochs=epoch, batch_size=batch_size)
        print(self.model.summary())
    def save_model(self):

        self.model.save('my_model.h5')

    def load_model(self):

        self.model=load_model('mymodel.h5')


    def predict(self, predict):

        result=self.model.predict(predict)
        print(result)


