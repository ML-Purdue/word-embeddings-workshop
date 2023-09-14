#!/usr/bin/env python3

from gensim.models import KeyedVectors
from IPython import embed
import const

if __name__ == '__main__':
    model = KeyedVectors.load_word2vec_format(const.PRETRAINED_MODEL_PATH, binary=True)

    model.most_similar('man')
    model.most_similar(model['king'] - model['man'] + model['woman'])
    embed()
