import csv
import os
import argparse

import numpy as np
import pandas as pd
import tqdm
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.disc_models.rnn.utils import SymbolTable
from lstm_utils import read_csv_file, read_word_dict

# Set up the parser
parser = argparse.ArgumentParser(description="This program is used to train an LSTM.")
subparsers = parser.add_subparsers(help="sub-command help", dest="command")

train_parser = subparsers.add_parser("train", help="Use this command to tell the program to train the LSTM.")
train_parser.add_argument("--train_data", help="Use this flag to specify where the train data can be found.")
train_parser.add_argument("--train_ends", help="Use this flag to specify where the train ends can be found.")
train_parser.add_argument("--train_marginal", help="Use this flag to specify where the train marginals can be found.")
train_parser.add_argument("--train_word_dict", help="Use this flag to specify where the training word dict can be found.")
train_parser.add_argument("--dev_data", help="Use this flag to specify where the development data can be found.")
train_parser.add_argument("--dev_labels", help="Use this flag to specify where the development labels for the dev data can be found.")
train_parser.add_argunment("--save_dir", help="Use this flag to specify the directory to save the trained model.")

evaluate_parser = subparsers.add_parser("evaluate", help="Use this command to tell the program to produce marginals from the LSTM.")
evaluate_parser.add_argument("--model", help="Use this flag to specify what lstm model to use.")
evaluate_parser.add_argument("--data", help="Use this flag to specify the data input data for the lstm.")
evaluate_parser.add_argument("--output", help="Use this flag to specify the output for the lstm.")

args = parser.parse_args()

# Train the LSTM
if args.command == "train":
    X_train, max_val = read_csv_file(args["train_data"], get_max_val=True)
    X_dev = read_csv_file(args["dev_data"], get_max_val=False)
    X_ends = read_csv_file(args["train_ends"], get_max_val=False)
    X_ends = np.array(map(lambda x: x[0], X_ends))
    train_marginals = np.loadtxt(args["train_marginals"])
    Y_dev = np.loadtxt(args["dev_labels"])
    word_dict = read_word_dict(args["train_word_dict"])

    train_marginals[train_marginals < 0] = 0
    train_kwargs = {
        'lr':         0.001,
        'dim':        100,
        'n_epochs':   10,
        'dropout':    0.5,
        'print_freq': 1,
        'max_sentence_length': 2000,
    }

    lstm = reRNN(seed=100, n_threads=20)
    lstm.word_dict = SymbolTable()
    lstm.word_dict.d = word_dict
    lstm.word_dict.s = max_val

    np.random.seed(200)
    training_size = int(len(X_train) * float(sys.argv[1]))
    train_idx = np.random.randint(0,len(X_train),size=training_size)
    lstm.train(X_train[train_idx], X_ends[train_idx], train_marginals[train_idx], X_dev=X_dev, Y_dev=Y_dev, save_dir='{}'.format(args["save_dir"]),  **train_kwargs)

# Evaluate the LSTM
if args.command == "evaluate":
    X_test = read_csv_file(args["data"], get_max_val=False)
    lstm = reRNN(seed=100, n_threads=10)
    lstm.load(save_dir=args["model"], model_name="reRNN")

    rnn_marginals = lstm.marginals(X_test, batch_size=1000)
    pd.DataFrame(rnn_marginals, columns=["RNN_marginals"]).to_csv(args["output"], index=False)
