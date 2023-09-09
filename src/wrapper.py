#!/usr/bin/env python3

from chromadb import Documents, EmbeddingFunction, Embeddings

class HuggingFaceWrapper(EmbeddingFunction):
    def __call__(self, texts: Documents) -> Embeddings:
        ...
        return embeddings
