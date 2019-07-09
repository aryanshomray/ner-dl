from tensorflow.keras.models import load_model
from spacy import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import itertools
nlp = load('en_vectors_web_lg')
a = []
class Fit:
    
    def __init__(self,line):
        self.line=line
        self.line_split=text_to_word_sequence(self.line, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,
                                              split=' ')
        self.word=[word for word in self.line_split if word.isalpha()]
        self.wordvec=[nlp(word).vector for word in self.line_split if word.isalpha()]
        self.pad_seq=pad_sequences(a.append(self.wordvec), maxlen=30, dtype='float32', padding='pre', truncating='pre',
                                   value=0.0)
        self.model=load_model('my_model.h5')
        self.seq_len=len(self.wordvec.shape[0])
        self.pre_dict={'O': [1,0,0,0,0,0,0,0,0], 'geo': [0,1,0,0,0,0,0,0,0], 'gpe': [0,0,1,0,0,0,0,0,0],
                       'per': [0,0,0,1,0,0,0,0,0], 'org': [0,0,0,0,1,0,0,0,0], 'nat': [0,0,0,0,0,1,0,0,0],
                       'eve': [0,0,0,0,0,0,1,0,0], 'art': [0,0,0,0,0,0,0,1,0], 'tim': [0,0,0,0,1,0,0,0,1]}
        self.dict={v: k for k, v in self.pre_dict.items()}
        self.answer=self.dict(np.argmax(self.model.predict(self.pad_seq),axis=2)[0])
        return dict(itertools.izip(self.word[-self.seq_len:],self.answer[-self.seq_len:]))



print(Fit('The strongest man on Earth is Mark Henry'))


