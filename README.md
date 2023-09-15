# Introduction to Word Embeddings

In this workshop, we will learn about word embeddings, train our own word embeddings, and then use the word embeddings for some cool applications.

## Setup 

### Local
From the root of the repository:
```bash
$ pip install -r requirements.txt ; dvc pull -r origin
```

### Colab
Open `notebook.ipynb` and press the 'Open in Colab' button, and you should be able to run the cells.

## Walkthrough
Once you have the environment setup, change your directory to `src/`. You can run all your code from here.

### Dataset

There are two ways Word2Vec uses contextual information to pre-process the dataset:
1. Continuous Bag-of-Words (CBOW): Here, we generate one-hot encode a target word as the output vector, and few-hot encode all the words adjacent to the target word.
2. Skip-Gram: Here, we one-hot encode a target word as the *input* vector, and for the output vector return each of the words adjacent to the target word encoded with a number that sums to one.

You can play around with either pre-processing technique by running:
```bash
$ python -m data.cbow # or
$ python -m data.skipgram
```

### Torch Word2Vec

Although the objective is not exactly the same, we use a feedforward autoencoder for the model. We are interested in the encoding weights of this architecture, since these are the actual word vectors that we desire.

You can play around with the untrained model by running:
```bash
$ python -m w2v.arch
```

Finally, you can execute the entire training sequence by running:
```bash
$ python -m w2v.train
```

You can modify hyperparameters by updating `src/const.py`. You can get an (ugly) graph of the embeddings if you set `const.VECTOR_DIMENSIONS = 2` and run `model.visualize(dataset.words)`

### Gensim

[Gensim](https://radimrehurek.com/gensim/intro.html) is a brilliant library that allows us to abstract implementation details for Word2Vec, with optimizations for both training and inference. 

You can run training code for Gensim using:
```bash
$ python -m gs.train
```

However, even this is dependent on good data, so we can skip that process by using Google's weights:
```bash
$ python -m gs.pretrained
```

Note: These pre-trained weights aren't that great either. Word embeddings generally aren't SoTA anymore. I recommend doing some research and finding the best consumer-grade solution available for your task, especially if you want to use embeddings in a production setting (as of 15 Sep 2023, [Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html) is a nice option).

----
