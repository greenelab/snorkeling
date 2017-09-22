import csv
import os

import numpy as np
import pandas as pd
import tqdm
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.disc_models.rnn.utils import SymbolTable

from utils import read_csv_file

X_test = read_csv_file("pmacs/test_candidates_offsets.csv", get_max_val=False)
Y_test = np.loadtxt("pmacs/test_candidates_labels.txt")

lstm = reRNN(seed=100, n_threads=4)
lstm.load(save_dir='checkpoints/one_percent/', model_name="reRNN")

rnn_marginals = lstm.marginals(X_test, batch_size=1000)
pd.DataFrame(rnn_marginals, columns=["RNN_marginals"]).to_csv("RNN_1_marginals.csv", index=False)
