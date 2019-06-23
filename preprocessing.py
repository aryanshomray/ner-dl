import spacy
nlp = spacy.load('en_core_web_lg')



class Process:

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
    def fit_data(self,data):
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
                    tags.append(line[3])
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
                    tags.append(line[3])
                    self.words.add(line[1].lower())
                    self.tags.add(line[3])

    def sent_vectorizer(self):


    def vector_gen(self):
        for i in self.words:
            self.word2vec[i]=nlp(i).vector
        sent_vectorizer()








