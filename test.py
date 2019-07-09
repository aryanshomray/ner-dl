from preprocessing import Process
from model import MyModel
print('Import Successful')
import spacy
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
nlp = spacy.load('en_core_web_sm')

process = Process()
process.fit_data('data/ner_dataset.csv')
print('Fitting Successful')

tensorboard=TensorBoard(log_dir='logs/{}'.format(time()))
process.vector_gen()
print('vec done')
process.padding()
print('Padding done')

model = MyModel(process.train)
model.model_compile()
print('Model Successfully Compiled')
model.training(process.train, process.labels)
print('Model trained Successfully')
model.save_model()
print('Model Saved')
