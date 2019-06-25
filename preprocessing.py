import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

nlp = spacy.load('en_core_web_sm')


class Process:

    nlp = spacy.load('en_core_web_sm')

    def __init__(self):
        self.trained_sentences=[]
        self.corr_tags=[]
        self.words=set()
        self.tags=set()
        self.vec_size=300
        self.max_length=20
        self.word2vec=dict()
        self.vec_sent=[]
        self.vec_tags=[]
        self.train=[]
        self.labels=[]
        self.tag_dict={'O':[1,0,0,0,0], 'geo':[0,1,0,0,0], 'gpe':[0,0,1,0,0], 'per':[0,0,0,1,0], 'org':[0,0,0,0,1]}
        self.batch_size=None

    def fit_data(self, data):
        '''This function is used to convert the raw data into sentences along with the tags for further training.'''
        sent=[]
        tags=[]
        with open(data,'r') as file:
            for l in file:
                line=l.split(',')
                if not line[0]:
                    if not line[1].isalpha():
                        continue
                    sent.append(line[1].lower())
                    if line[3]=='O':
                        tags.append(line[3])
                    else:
                        tags.append(line[3][2:])
                    self.words.add(line[1].lower())
                    self.tags.add(line[3])
                else:
                    self.trained_sentences.append(sent)
                    self.corr_tags.append(tags)
                    sent = []
                    tags = []
                    if not line[1].isalpha():
                        continue
                    sent.append(line[1].lower())
                    if line[3] == 'O':
                        tags.append(line[3])
                    else:
                        tags.append(line[3][2:])
                    self.words.add(line[1].lower())
                    self.tags.add(line[3])

    def vector_gen(self):
        for word in self.words:
            self.word2vec[word]=nlp(word).vector
        for sen in self.trained_sentences:
            sent=[]
            for word in sen:
                sent.append(self.word2vec[word])
            self.vec_sent.append(sent)
        for sen_tag in self.corr_tags:
            tags = []
            for tag in sen_tag:
                tags.append(self.tag_dict[tag])
            self.vec_tags.append(tags)

    def padding(self):
        self.train=pad_sequences(self.vec_sent, maxlen=30, dtype='int32', padding='pre', truncating='pre', value=0.0)
        self.labels=pad_sequences(self.vec_tags, maxlen=30, dtype='int32', padding='pre', truncating='pre', value=0.0)

    def test_prep(self,line):

        line=tf.keras.preprocessing.text.text_to_word_sequence(line, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                                               lower=True, split=' ')
        line = [nlp(word).vector for word in line if not word.isalpha()]
        line = pad_sequences(line, maxlen=30, dtype='int32', padding='pre', truncating='pre', value=0.0)
        return line
