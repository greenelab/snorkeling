from collections import OrderedDict, defaultdict
import re
import operator
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from tqdm import tqdm_notebook
import torch
from snorkel.labeling import LabelModel
from .dataframe_helper import generate_results_df

def indexed_combination(seq, n):
    # obtained from 
    # https://stackoverflow.com/questions/47234958/generate-a-random-equally-probable-combination-from-a-list-in-python
    result = []
    for u in seq:
        if n & 1:
            result.append(u)
        n >>= 1
        if not n:
            break
    return result


def sample_lfs(lf_list, size_of_sample_pool, size_per_sample, number_of_samples, random_state=100):
    pd.np.random.seed(random_state)
    bit_list = [1 if i < size_per_sample else 0 for i in range(size_of_sample_pool)]
    already_seen = set({})
    lf_combinations = []

    for sample in range(number_of_samples):
        # sample with replacement
        pd.np.random.shuffle(bit_list)

        # obtained from
        # https://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
        out=0
        for bit in bit_list:
            out = (out << 1) | bit

        lf_combinations.append(indexed_combination(lf_list, out))

    return lf_combinations

def train_model_random_lfs(randomly_sampled_lfs, train_matrix, dev_matrix, dev_labels, test_matrix, regularization_grid):
    hyper_grid_results = defaultdict(dict)
    train_grid_results = defaultdict(dict)
    dev_grid_results = defaultdict(dict)
    test_grid_results = defaultdict(dict)
    models = defaultdict(dict)
    
    for lf_sample in tqdm_notebook(enumerate(randomly_sampled_lfs)):
        for param in regularization_grid:

            label_model=LabelModel(cardinality=2)
            label_model.fit(
                train_matrix[:,lf_sample[1]], n_epochs=1000, 
                seed=100, lr=0.01, l2=param,
            )
            
            # Get marginals for each parameter
            hyper_grid_results[str(param)] = roc_curve(
                dev_labels, 
                label_model.predict_proba(dev_matrix[:,lf_sample[1]])[:,1]
            )
            
        # Convert marginals into AUROCs
        hyper_grid_results = {
            param:auc(hyper_grid_results[param][0], hyper_grid_results[param][1])
            for param in hyper_grid_results
        }

        # Select the parameter with the highest AUROC
        best_param = float(max(hyper_grid_results.items(), key=operator.itemgetter(1))[0])

        # Re-fit the model
        label_model.fit(
                train_matrix[:,lf_sample[1]], n_epochs=1000, 
                seed=100, lr=0.01, l2=best_param,
        )
        
        # Save marginals for output
        key = f'{lf_sample[0]}:{",".join(map(str, lf_sample[1]))}'
        train_grid_results[key] = label_model.predict_proba(train_matrix[:,lf_sample[1]])
        dev_grid_results[key] = label_model.predict_proba(dev_matrix[:,lf_sample[1]])
        test_grid_results[key] = label_model.predict_proba(test_matrix[:,lf_sample[1]])
        models[key] = label_model
    
    return train_grid_results, dev_grid_results, test_grid_results, models

def get_model_performance(gold_labels, result_grid, num_lfs=1):
    records = []
    
    for key in result_grid:
        precision, recall, _ = precision_recall_curve(
            gold_labels, 
            result_grid[key][:,1]
        )
        fp, tp, _ = roc_curve(
            gold_labels,
            result_grid[key][:,1]
        )
        records.append({
            "aupr":auc(recall, precision),
            "auroc":auc(fp,tp),
            "lf_sample": key,
            "lf_num":num_lfs
        }) 
        
    return pd.DataFrame.from_records(records)
