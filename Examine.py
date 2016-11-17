import nltk
from nltk.corpus import conll2000

from UnigramChunker import UnigramChunker

chunked_corpus=conll2000.chunked_sents("train.txt",chunk_types=["NP"],tagset="universal")

chunker=UnigramChunker()
chunker.train(chunked_corpus,format_flag=True)
score=chunker.evaluate(chunked_corpus)
print(score)