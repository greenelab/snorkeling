from collections import OrderedDict, defaultdict
import re
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from tqdm import tqdm_notebook
import torch

from snorkel.learning import GenerativeModel
from metal.label_model import LabelModel

from .dataframe_helper import generate_results_df


def train_generative_model(data_matrix, burn_in=10, epochs=100, reg_param=1e-6, 
    step_size=0.001, deps=[], lf_propensity=False):
    """
    This function is desgned to train the generative model
    
    data_matrix - the label function matrix which contains the output of all label functions
    burnin - number of burn in iterations
    epochs - number of epochs to train the model
    reg_param - how much regularization is needed for the model
    step_size - how much of the gradient will be used during training
    deps - add dependencey structure if necessary
    lf_propensity - boolean variable to determine if model should model the likelihood of a label function
    
    return a fully trained model
    """
    model = GenerativeModel(lf_propensity=lf_propensity)
    model.train(
        data_matrix, epochs=epochs,
        burn_in=burn_in, reg_param=reg_param, 
        step_size=step_size, reg_type=2
    )
    return model

def run_grid_search(model, data,  grid, labels, cv=10):
    """
    This function is designed to find the best hyperparameters for a machine learning model.

    model - Sklearn model to be optimized
    data - the data to train the model
    grid - the search grid for each model
    labels - binary training labels for optimization criteria
    cv - the number of cross validation folds to use
    """

    searcher = GridSearchCV(model, param_grid=grid, cv=cv, return_train_score=True, scoring=['roc_auc', 'f1'], refit='roc_auc')
    return searcher.fit(data, labels)

def get_attn_scores(model_paths, end_model,  datapoint, words):
    """
    This function extracts the attention layer scores from the LSTM or GRU model.
    
    model_paths - the paths to the lstm models saved in pytorch format
    end_model - the end model object need to load the pytorch weights
    datapoint - the datapoint to extract the attn scores
    words - the list of words the datapoint contains
    """
    attn_df_dict = {}
    for model_params in tqdm_notebook(model_paths.keys()):
        if model_params not in end_model:
            continue
        attn_scores = []
        for model in model_paths[model_params]:
            params = torch.load(model, map_location="cpu")
            fixed_model = OrderedDict([(re.sub(r'module\.', '', key), data) for key, data in params['model'].items()])
            end_model[model_params].load_state_dict(fixed_model)
            end_model[model_params].eval()
            end_model[model_params].predict_proba(datapoint)
            attn_scores.append(end_model[model_params].network[1][0].attn_score.detach().numpy().flatten())
        
        attn_df = pd.DataFrame(
            pd.np.stack(attn_scores, axis=1),
            columns=["attn_score_epoch_{}".format(epoch) for epoch in range(len(attn_scores))]
        )
        attn_df['words'] = words
        attn_df_dict[model_params] = attn_df

    return attn_df_dict


def get_network_results(model_paths, end_model, dev_X, test_X):
    """
    This function calculates the validation loss for the model
    
    model_paths - the paths to the models saved in pytorch format
    end_model - the end model object need to load the pytorch weights
    X - the data for each model to evaulate
    Y - the labels for the loss to evaluate
    loss_fn - the loss function 
    """
    learning_rate = []
    train_loss = []
    val_loss = []
    acc_score = []
    for model in tqdm_notebook(model_paths):
        params = torch.load(model, map_location="cpu")
        learning_rate.append(params['optimizer']['param_groups'][0]['lr'])
        train_loss.append(params['train_loss'])
        val_loss.append(params['val_loss'])
        acc_score.append(params['score'])

    params = torch.load(os.path.dirname(model)+"/best_model.pth", map_location="cpu")
    fixed_model = OrderedDict([(re.sub(r'module\.', '', key), data) for key, data in params['model'].items()])
    end_model.load_state_dict(fixed_model)
    end_model.eval()

    loss_df = (
        pd.DataFrame(
            pd.np.stack([
                list(range(len(train_loss))), train_loss, 
                val_loss, learning_rate,
                acc_score
            ], axis=1), 
            columns = ["epoch", "train_loss", "val_loss", "lr", "acc"]
        )
    )

    dev_predictions_df = (
        pd.Series(
           end_model.predict_proba(dev_X)[:,0]
        )
    )
    test_predictions_df = (
        pd.Series(
            end_model.predict_proba(test_X)[:,0]
        )
    )
    return loss_df, dev_predictions_df, test_predictions_df, params['best_iteration']


def train_model_random_lfs(randomly_sampled_lfs, train_matrix, dev_matrix, dev_labels, test_matrix, regularization_grid):
    hyper_grid_results = defaultdict(dict)
    train_grid_results = defaultdict(dict)
    dev_grid_results = defaultdict(dict)
    test_grid_results = defaultdict(dict)
    for lf_sample in tqdm_notebook(enumerate(randomly_sampled_lfs)):
        for param in regularization_grid:

            label_model = LabelModel(k=2)
            label_model.train_model(
                train_matrix[:,lf_sample[1]], n_epochs=1000, 
                log_train_every=200, seed=100, lr=0.01, l2=param,
                verbose=False
            )
            
            hyper_grid_results[str(param)] = label_model.predict_proba(dev_matrix[:,lf_sample[1]])
       
        best_param = float(max(hyper_grid_results))
        label_model.train_model(
                train_matrix[:,lf_sample[1]], n_epochs=1000, 
                log_train_every=200, seed=50, lr=0.01, l2=best_param,
                verbose=False
        )
        
        key = f'{lf_sample[0]}:{",".join(map(str, lf_sample[1]))}'
        train_grid_results[key] = label_model.predict_proba(train_matrix[:,lf_sample[1]])
        dev_grid_results[key] = label_model.predict_proba(dev_matrix[:,lf_sample[1]])
        test_grid_results[key] = label_model.predict_proba(test_matrix[:,lf_sample[1]])
        
    return train_grid_results, dev_grid_results, test_grid_results

def train_baseline_model(
    train_matrix, 
    dev_matrix,
    dev_labels,
    test_matrix, 
    lf_indicies, 
    regularization_grid, 
    train_marginal_dir,
    write_file=False
):
    grid_results = {}
    dev_grid_results = {}
    test_grid_results = {}
    for param in regularization_grid:
        label_model = LabelModel(k=2)
        label_model.train_model(
            train_matrix[:,lf_indicies], n_epochs=1000, 
            log_train_every=200, seed=100, lr=0.01, l2=param,
            verbose=False, #Y_dev=dev_labels
        )
        grid_results[str(param)] = label_model.predict_proba(dev_matrix[:,lf_indicies])

    best_param = float(max(grid_results))
    label_model.train_model(
            train_matrix[:,lf_indicies], n_epochs=1000, 
            log_train_every=200, seed=50, lr=0.01, l2=best_param,
            verbose=False, #Y_dev=dev_labels
    )
    if write_file:
        (
            pd.DataFrame(
                label_model.predict_proba(train_matrix[:,lf_indicies]), 
                columns=["pos_class_marginals", "neg_class_marginals"]
            )
            .to_csv(f"{train_marginal_dir}baseline_marginals.tsv.xz", compression="xz", index=False, sep="\t")
        )

    dev_grid_results[best_param] = label_model.predict_proba(dev_matrix[:,lf_indicies])
    test_grid_results[best_param] = label_model.predict_proba(test_matrix[:,lf_indicies])

    return dev_grid_results, test_grid_results

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


def sample_lfs(lf_list, size_of_sample_pool, sample_size, number_of_samples, random_state=100):
    pd.np.random.seed(random_state)
    bit_list = [1 if i < sample_size else 0 for i in range(size_of_sample_pool)]
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


def run_random_additional_lfs(
    range_of_sample_sizes, range_of_lf_indicies,
    size_of_sample_pool, num_of_samples,
    train, dev, dev_labels,
    test, test_labels, grid,
    label_matricies,
    train_marginal_dir="",
    ds_start=0, ds_end=8
):

    """
    This function is designed to randomly sample label functions and
    traina generative model based on the sampled labeled functions.
    """
    dev_result_df = pd.DataFrame([], columns=["AUPRC", "AUROC", "num_lfs"])
    test_result_df = pd.DataFrame([], columns=["AUPRC", "AUROC", "num_lfs"])
    dev_marginals_df = pd.DataFrame([], columns=["marginals", "label", "num_lfs"])
    test_marginals_df = pd.DataFrame([], columns=["marginals", "label", "num_lfs"])
    lf_sample_keeper = {}

    for sample_size in range_of_sample_sizes:

        # Uniformly sample lfs to be added towards the set of
        # distant supervision lfs
        lf_samples = sample_lfs(
            range_of_lf_indicies,
            size_of_sample_pool,
            sample_size,
            num_of_samples
        )

        # add additional lfs to the distant supervision list
        lf_samples = list(map(lambda x: list(range(ds_start, ds_end+1)) + x, lf_samples))

        # Keep track of sampled labels
        lf_sample_keeper[sample_size] = lf_samples

        grid_results = train_model_random_lfs(
            lf_samples, train,
            dev,  dev_labels, test,
            grid
        )

        # Save the training marginals for each generative model run
        # This will be used for the discriminator model later
        (
            pd.DataFrame
            .from_dict({
                key: grid_results[0][key][:, 0]
                for key in grid_results[0]
            })
            .assign(
                candidate_id=(
                    label_matricies
                    .sort_values("candidate_id")
                    .candidate_id
                    .values
                )
            )
            .to_csv(
                f"{train_marginal_dir}/{sample_size}_sampled_lfs.tsv.xz",
                sep="\t", index=False, compression='xz'
            )
        )
        
        dev_marginals_df = dev_marginals_df.append(
            pd.DataFrame(
                pd.np.concatenate([list(zip(grid_results[1][key][:,0], dev_labels)) for key in grid_results[1]]),
                columns=["marginals", "label"]
            )
            .assign(num_lfs=sample_size),
            sort=True
        )
        
        test_marginals_df = test_marginals_df.append(
            pd.DataFrame(
                pd.np.concatenate([list(zip(grid_results[1][key][:,0], dev_labels)) for key in grid_results[2]]),
                columns=["marginals", "label"]
            )
            .assign(num_lfs=sample_size),
            sort=True
        )
        
        # Get the development set results
        dev_result_df = dev_result_df.append(
            generate_results_df(
                grid_results[1],
                dev_labels
            )
            .rename(index=str, columns={0: "AUPRC", 1: "AUROC"})
            .assign(num_lfs=sample_size)
            .reset_index()
            .drop("index", axis=1)
        )

        # Get the test set results
        test_result_df = test_result_df.append(
            generate_results_df(
                grid_results[2],
                test_labels
            )
            .rename(index=str, columns={0: "AUPRC", 1: "AUROC"})
            .assign(num_lfs=sample_size)
            .reset_index()
            .drop("index", axis=1)
        )

    return lf_sample_keeper, dev_result_df, test_result_df, dev_marginals_df, test_marginals_df
