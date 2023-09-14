#!/usr/bin/env python3

from IPython import embed
import pandas as pd
import torch
import const

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # load corpus
        df = pd.read_csv(path)

        # get word map for tensor conversion
        self.words = list(set(' '.join(list(df[const.COLUMN_NAME])).split()))
        self.nearby_words = dict.fromkeys(self.words, set())

        # preprocess corpus
        for sentence in list(df[const.COLUMN_NAME]):
            words = sentence.split()
            for idx, word in enumerate(words):
                nearby = set()
                if idx > 0: nearby = nearby.union({words[idx-1]})
                if idx != len(words)-1: nearby = nearby.union({words[idx+1]})
                
                self.nearby_words[word] = self.nearby_words[word].union(nearby)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # initialize tensors
        X = torch.zeros(len(self))
        y = torch.zeros(len(self))

        # one-hot encode tensors using skip-gram
        X[idx] = 1
        nearby = self.nearby_words[self.words[idx]]
        n_nearby = len(nearby)
        for word in nearby: y[self.words.index(word)] = 1 / n_nearby

        return X, y

if __name__ == '__main__':
    dataset = Dataset(const.TOY_DATASET_PATH)
    print(dataset[0])
    embed()
