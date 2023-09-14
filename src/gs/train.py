#!/usr/bin/env python3

from gensim.models import Word2Vec
from IPython import embed
import pandas as pd
import const
import nltk

if __name__ == '__main__':
    nltk.download('punkt')

    df = pd.read_csv(const.DATASET_PATH)
    data = df[const.COLUMN_NAME].apply(nltk.word_tokenize)

    model = Word2Vec(data,
                     min_count=1,
                     vector_size=const.VECTOR_DIMENSIONS)

    model.wv.most_similar('man')
    model.save(str(const.SAVE_MODEL_PATH / 'trained-gensim.kv'))

    embed()
    model.wv.most_similar(model.wv['king'] + model.wv['man'] + model.wv['woman'])
