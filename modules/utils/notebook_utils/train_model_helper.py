from collections import OrderedDict
import re
import os

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from tqdm import tqdm_notebook
import torch

from snorkel.learning import GenerativeModel

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