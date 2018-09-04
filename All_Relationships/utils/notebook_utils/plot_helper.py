import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_cand_histogram(model_names, lfs_columns, data_df, plot_title, xlabel):
    """
    This function is designed to plot a histogram of the candiadte marginals

    model_names - labels for each model
    lfs_columns - a listing of column indexes that correspond to desired label fucntions
    data_df - the dataframe that contains the marginals
    plot_title - the title of the plot
    xlabel - the title of the xaxis
    """
    f, axs = plt.subplots(len(lfs_columns), 1, figsize=(10, 7), sharex=True)
    if not isinstance(axs, list):
        axs = [axs]

    for columns, ax in zip(model_names, axs):
        sns.distplot(data_df[columns], ax=ax, kde=False, axlabel=False)
        ax.set_ylabel(columns, fontsize=12)

    f.suptitle(plot_title)
    plt.xlabel(xlabel, fontsize=12)
    plt.show()
    return

def plot_roc_curve(marginals_df, true_labels, model_names, plot_title):
    """
    This function is designed to plot ROC curves for the models.

    marginals_df - a dataframe containing marginals from the generative model
    true_labels - a list of true labels
    model_names - labels for each model
    plot_title - the title of the plot
    """
    plt.figure(figsize=(10,6))
    plt.plot([0,1], [0,1], linestyle='--', color='grey', label="Random (AUC = 0.50)")

    for marginal, model_label in zip(marginals_df.columns, model_names):
        fpr, tpr, threshold = roc_curve(true_labels, marginals_df[marginal])
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=model_label+" (AUC = {:0.2f})".format(area))

    plt.title(plot_title)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    return 

def plot_pr_curve(marginals_df, true_labels, model_names, plot_title):
    """
    This function is designed to plot PR curves for the models.

    marginals_df - a dataframe containing marginals from the generative model
    true_labels - a list of true labels
    model_names - labels for each model
    plot_title - the title of the plot
    """
    plt.figure(figsize=(10,6))
    positive_class = true_labels.sum()/len(true_labels)
    plt.plot([0,1], [positive_class, positive_class], color='grey', 
             linestyle='--', label='Baseline (AUC = {:0.2f})'.format(positive_class))

    for marginal, model_label in zip(marginals_df.columns, model_names):
        precision, recall, threshold = precision_recall_curve(true_labels, marginals_df[marginal])
        area = auc(recall, precision)
        plt.plot(recall, precision, label=model_label+" (AUC = {:0.2f})".format(area))

    plt.title(plot_title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    return 

