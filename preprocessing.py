import pandas as pd

class Process:
    def __init__(self, data):
        self.max_sent_size=100
        self.vec_size=100
        self.data=data
        self.words=set()
        self.tags=set()
        self.sentences=[]
        self.tag=[]
        self.x=[]
        self.y=[]

    def get_data(self):
        words=[]
        tags=[]
        df=pd.read_csv(self.data,header=None,index=None)
        for index,rows in df.iterrows():
            if not rows[0]:
                words.append(rows[1])
                tags.append(rows[3])
                self.words.add(rows[1])
                self.words.add(rows[3])
            elif len(words) and len(tags):
                    self.sentences.append(words)
                    self.tag.append(tags)
                    words = [rows[1]]
                    tags = [rows[3]]
