#!/usr/bin/env python3

import matplotlib.pyplot as plt
from IPython import embed
import torch
import const


class Word2Vec(torch.nn.Module):
    def __init__(self, input_shape, vector_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_shape, vector_size)
        self.linear2 = torch.nn.Linear(vector_size, input_shape)

        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.softmax(x)

    def visualize(self, vocabulary):
        x, y = self.embedding.to('cpu').detach().numpy()

        plt.scatter(x, y)
        for idx, word in enumerate(vocabulary): plt.annotate(word, (x[idx], y[idx]))
        plt.show()

    @property
    def embedding(self):
        return self.get_submodule('linear1').weight


if __name__ == '__main__':
    model = Word2Vec(4, const.VECTOR_DIMENSIONS)

    print(model)
    print(model(torch.tensor([0., 0., 1., 1.])))
    embed()
