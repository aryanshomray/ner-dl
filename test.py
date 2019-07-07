from preprocessing import Process
from model import MyModel
print('Import Successful')
import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

nlp = spacy.load('en_core_web_sm')

process = Process()
process.fit_data('data/ner_dataset.csv')
print('Fitting Successful')


process.vector_gen()
print('vec done')
process.padding()
print('Padding done')

model = MyModel(process.train)
model.model_compile()
print('Model Successfully Compiled')
model.training(process.train,process.labels,epoch=3)
print('Model trained Successfully')
model.save_model()
print('Model Saved')
