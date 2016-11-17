import nltk
from nltk.corpus import conll2000
import numpy as np

def preprocess(chunked_corpus):
    corpus=[]
    for tree in chunked_corpus:
        sent=nltk.tree2conlltags(tree)
        corpus.append([(p,c) for w,p,c in sent])
    return corpus
def distribution(data):
    count_pc=nltk.defaultdict(lambda :0)
    POS_labels=set()
    IOB_labels=set()
    for sent in data:
        for p,c in sent:
            pc=p+"_"+c
            count_pc[pc]+=1
            POS_labels.add(p)
            IOB_labels.add(c)
    return count_pc,list(POS_labels),list(IOB_labels)


def extract_pos_corpus(trees):
    pos_corpus=[]
    for tree in trees:
        iob_format=nltk.chunk.tree2conlltags(tree)
        pos_sent=[(w,p) for w,p,c in iob_format]
        pos_corpus.append(pos_sent)
    return pos_corpus

class UnigramChunker():
    def __init__(self,corpus=None):
        self.count_=nltk.defaultdict(lambda :0)
        self.set_POS=[]
        self.set_IOB=[]
    def train(self,corpus,format_flag=False):
        my_corpus=corpus
        if format_flag==True:
            my_corpus=preprocess(corpus)

        self.count_,self.set_POS,self.set_IOB=distribution(my_corpus)
    def predict(self,sent):
        pos_sent=[p for w,p in sent ]

        prediction=[]

        for p in pos_sent:
            score=[]
            choices=[]
            for c in self.set_IOB:
                pc=p+"_"+c
                score.append(self.count_[pc])
                choices.append(c)
            prediction.append(choices[np.argmax(score)])

        conlltag_prediction=[(w,p,c) for ((w,p),c) in zip(sent,prediction)]
        return nltk.chunk.conlltags2tree(conlltag_prediction)

    def predict_many(self,sents):
        predictions=[]
        for sent in sents:
            predictions.append(self.predict(sent))
        return predictions

    def evaluate(self,chunked_trees):
        chunk_score=nltk.chunk.ChunkScore()

        pos_corpus=extract_pos_corpus(chunked_trees)
        predictions=self.predict_many(pos_corpus)

        for iter,tree in enumerate(chunked_trees):
            chunk_score.score(tree,predictions[iter])

        return chunk_score




