#!/usr/bin/env python3

from IPython import embed
import pandas as pd
import torch
import const


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # load corpus
        df = pd.read_csv(path)
        df[const.COLUMN_NAME] = df[const.COLUMN_NAME].apply(lambda x: x.lower())  # uncased

        # get word map for tensor conversion
        self.words = list(set(' '.join(list(df[const.COLUMN_NAME])).split()))

        # preprocess corpus
        self.data = []
        for sentence in list(df[const.COLUMN_NAME]):
            words = sentence.split()
            for idx, word in enumerate(words):
                nearby = set()
                if idx > 0: nearby = nearby.union({words[idx-1]})
                if idx != len(words)-1: nearby = nearby.union({words[idx+1]})
                self.data.append((word, nearby))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # initialize tensors
        X = torch.zeros(len(self))
        y = torch.zeros(len(self))

        # few-hot encode tensors using continuous bag of words
        y[self.words.index(self.data[idx][0])] = 1
        nearby = self.data[idx][1]
        for word in nearby: X[self.words.index(word)] = 1

        return X, y


if __name__ == '__main__':
    dataset = Dataset(const.DATASET_PATH)
    print(dataset[0])
    embed()
