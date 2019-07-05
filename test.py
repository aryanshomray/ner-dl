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
print((process.corr_tags))
print((process.tags))
print(len(process.tags))
a=[]
for i in process.trained_sentences:
    a.append(len(i))
print(a)
b=[]
for i in process.corr_tags:
    b.append(len(i))
print(b)

for i in range(len(a)):
    if a[i]!=b[i]:
        print('Fail')

process.vector_gen()
print('vec done')
process.padding()
print('Padding done')

process.save_data()