import tensorflow as tf
import numpy as np


class MyModel:

    def __init__(self,train,test,vec_size=300):

        self.train=np.array(train)
        self.test=np.array(test)
        self.vec_size=vec_size
        self.model=None
        self.sent_length=20
        self.testing=None

    def model_compile(self):

        self.model=tf.keras.Sequential()

        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=5,
                                                                          activation='softmax',
                                                                          input_shape=(self.train.shape[0],
                                                                                       self.train.shape[1],
                                                                                       self.train.shape[2]),
                                                                          return_sequences=True)))

        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    def train(self,epoch=10,batch_size=64):

        self.model.fit(self.train,self.train,epochs=epoch,batch_size=batch_size)

    def save_model(self):
        self.model.save('my_model.h5')

    def load_model(self):
        self.model=tf.keras.models.load_model('mymodel.h5')

    def predict(self,self.testing):
        result=self.model.predict()

