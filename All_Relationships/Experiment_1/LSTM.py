import csv
import os

import numpy as np
import tqdm
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.disc_models.rnn.utils import SymbolTable

from utils import read_csv_file, read_word_dict

X_train, max_val = read_csv_file("pmacs/train_candidates_offsets.csv", get_max_val=True)
X_ends = read_csv_file("pmacs/train_candidates_ends.csv", get_max_val=False)
X_ends = np.array(map(lambda x: x[0], X_ends))
X_dev = read_csv_file("pmacs/dev_candidates_offsets.csv", get_max_val=False)
train_marginals = np.loadtxt("pmacs/train_marginals")
Y_dev = np.loadtxt("pmacs/dev_candidates_label.txt")
word_dict = read_word_dict("pmacs/train_word_dict.csv")

train_kwargs = {
    'lr':         0.001,
    'dim':        100,
    'n_epochs':   10,
    'dropout':    0.5,
    'print_freq': 1,
    'max_sentence_length': 2000,
}

lstm = reRNN(seed=100, n_threads=4)
lstm.word_dict = SymbolTable()
lstm.word_dict.d = word_dict
lstm.word_dict.s = max_val

np.random.seed(200)
training_size = int(len(X_train) * 0.1)
train_idx = np.random.randint(0, len(X_train), size=training_size)
lstm.train(X_train[train_idx], X_ends[train_idx], train_marginals[train_idx], X_dev=X_dev, Y_dev=Y_dev, save_dir='ten_percent',  **train_kwargs)
