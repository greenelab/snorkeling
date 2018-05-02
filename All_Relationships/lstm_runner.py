import csv
import os
import argparse

import numpy as np
import pandas as pd
import tqdm
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.disc_models.rnn.utils import SymbolTable

def read_word_dict(filename):
    """
     Read a CSV into a dictionary using the Key column (as string) and Value column (as int).
    Keywords:
    fielname - name of the file to read
    """
    data = {}
    with open(filename, 'rb') as f:
        input_file = csv.DictReader(f)
        for row in tqdm.tqdm(input_file):
            data[row['Key']] = int(row['Value'])
    return data

# Set up the parser
parser = argparse.ArgumentParser(description="This program is used to train an LSTM.")
subparsers = parser.add_subparsers(help="sub-command help", dest="command")

train_parser = subparsers.add_parser("train", help="Use this command to tell the program to train the LSTM.")
train_parser.add_argument("--train_data", help="Use this flag to specify where the train data can be found.")
train_parser.add_argument("--train_word_dict", help="Use this flag to specify where the training word dict can be found.")
train_parser.add_argument("--dev_data", help="Use this flag to specify where the development data can be found.")
train_parser.add_argument("--save_dir", help="Use this flag to specify the directory to save the trained model.")
train_parser.add_argument("--lr", help="Use this flag to specify the learning rates of LSTM model.", nargs="+", type=float)
train_parser.add_argument("--dropout", help="Use this flag to specify the dropout rate of LSTM model.", nargs="+", type=float)
train_parser.add_argument("--dim", help="Use this flag to specify the dimensions of the hidden state of the LSTM model.",type=int,  nargs="+")
train_parser.add_argument("--epochs", help="Use this flag to specify the number of epochs to train the LSTM model.",type=int,  default = 100)

evaluate_parser = subparsers.add_parser("evaluate", help="Use this command to tell the program to produce marginals from the LSTM.")
evaluate_parser.add_argument("--model", help="Use this flag to specify what lstm model to use.")
evaluate_parser.add_argument("--data", help="Use this flag to specify the data input data for the lstm.")
evaluate_parser.add_argument("--output", help="Use this flag to specify the output for the lstm.")

args = parser.parse_args()

# Train the LSTM
if args.command == "train":
    np.random.seed(100)
    X_train_df = pd.read_csv(args.train_data, sep="\t")
    
    X_train = [map(int, x.split(",")) for x in X_train_df.data_str]
    
    max_val = max(map(lambda x: max(x), X_train))
    
    X_ends = X_train_df.ends.values
    train_marginals = X_train_df.label.values
    
    word_dict = read_word_dict(args.train_word_dict)
    
    if args.dev_data != None:
        X_dev_df = pd.read_csv(args.dev_data, sep="\t")
        X_dev = [map(int, x.split(",")) for x in X_dev_df.data_str]
        Y_dev = X_dev_df.label.values
    else:
        X_dev = None
        Y_dev = None

    lstm = reRNN(seed=100, n_threads=20)
    lstm.word_dict = SymbolTable()
    lstm.word_dict.d = word_dict
    lstm.word_dict.s = max_val
    
    for lr in args.lr:
        for dropout in args.dropout:
            for dim in args.dim:
                train_kwargs = {
                    'lr':         lr,
                    'dim':        dim,
                    'n_epochs':   args.epochs,
                    'dropout':    dropout,
                    'print_freq': 1,
                    'max_sentence_length': 962,
                    'batch_size':200
                }
                
                lstm.train(
                    X_train, X_ends, 
                    train_marginals, 
                    X_dev=X_dev, Y_dev=Y_dev, 
                    save_dir='{}'.format(args.save_dir), 
                    filename=(
                        "experiments/snorkel_trial/lstm_{}_{}_{}.tsv"
                        .format(lr, dropout,dim)
                        ),
                    dev_ckpt=False,
                    **train_kwargs)
                lstm.save(save_dir=args.save_dir)

# Evaluate the LSTM
if args.command == "evaluate":
    X_test_df = pd.read_csv(args.data)
    X_test = [map(int, x.split(",")) for x in X_test_df.data_str]
    
    lstm = reRNN(seed=100, n_threads=10)
    lstm.load(save_dir=args.model, model_name="reRNN")

    rnn_marginals = lstm.marginals(X_test, batch_size=1000)
    pd.DataFrame(rnn_marginals, columns=["RNN_marginals"]).to_csv(args.output, index=False)
