#!/usr/bin/env python3

from pathlib import Path

# data
COLUMN_NAME = 'text'
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
DATASET_PATH = DATA_DIR / 'corpus.csv'

# architecture
VECTOR_DIMENSIONS = 2

# training
EPOCHS = 1000
MOMENTUM = .9
BATCH_SIZE = 10
LEARNING_RATE = 1E-4

# model persistence
SAVE_MODEL_PATH = BASE_DIR / 'models'
PRETRAINED_MODEL_PATH = SAVE_MODEL_PATH / 'GoogleNews-vectors-negative300.bin.gz'
