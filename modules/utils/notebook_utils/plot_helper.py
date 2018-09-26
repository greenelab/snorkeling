import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import scipy.sparse as sparse

def plot_cand_histogram(model_names, lfs_columns, data_df, plot_title, xlabel):
    """
    This function is designed to plot a histogram of the candiadte marginals

    model_names - labels for each model
    lfs_columns - a listing of column indexes that correspond to desired label fucntions
    data_df - the dataframe that contains the marginals
    plot_title - the title of the plot
    xlabel - the title of the xaxis
    """
    f, axs = plt.subplots(len(lfs_columns), 1, figsize=(10, 10), sharex=True)
    if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
        axs = [axs]

    for columns, ax in zip(model_names, axs):
        sns.distplot(data_df[columns], ax=ax, kde=False, axlabel=False)
        ax.set_ylabel(columns, fontsize=12)

    f.suptitle(plot_title)
    plt.xlabel(xlabel, fontsize=12)
    plt.show()
    return

def plot_roc_curve(marginals_df, true_labels, plot_title="ROC", model_type='scatterplot', xlim=[0,1], figsize=(10,6)):
    """
    This function is designed to plot ROC curves for the models.

    marginals_df - a dataframe containing marginals from the generative model
    true_labels - a list of true labels
    model_names - labels for each model
    plot_title - the title of the plot
    """
    
    model_aucs = {}
    plt.figure(figsize=figsize)
    
    #Get marginals
    model_roc_rates = {
        model:roc_curve(true_labels, marginals_df[model])
        for model in marginals_df.columns
    }
    model_aucs = {
        model:auc(model_roc_rates[model][0], model_roc_rates[model][1]) 
        for model in model_roc_rates
    }
            
    if model_type == 'barplot':
        display_df = (
            pd.DataFrame
            .from_dict(model_aucs,orient='index', columns=["auc"])
            .reset_index()
            .rename(index=str, columns={"index": "model"})
        )
        sns.barplot(x='auc', y='model', data=display_df, color='blue')
        plt.xlim(xlim)
        plt.title(plot_title)
    elif model_type == 'curve':
        plt.plot([0,1], [0,1], linestyle='--', color='grey', label="Random (AUC = 0.50)")
        
        for model in model_roc_rates:
            plt.plot(
                model_roc_rates[model][0],
                model_roc_rates[model][1], 
                label=model+" (AUC = {:0.2f})".format(model_aucs[model])
            )

        plt.title(plot_title)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
    elif model_type == 'scatterplot':
        display_df = (
            pd.DataFrame
            .from_dict(model_aucs,orient='index', columns=["auc"])
            .reset_index()
            .rename(index=str, columns={"index": "model"})
        )
        sns.pointplot(x='model', y='auc', data=display_df)
        plt.title(plot_title)
    else:
        raise Exception("Please pick a valid option: barplot, curve, scatterplot")
    return model_aucs

def plot_pr_curve(marginals_df, true_labels, plot_title="PRC", model_type="barplot", xlim=[0,1], figsize=(10,6)):
    """
    This function is designed to plot PR curves for the models.

    marginals_df - a dataframe containing marginals from the generative model
    true_labels - a list of true labels
    model_names - labels for each model
    plot_title - the title of the plot
    """
    model_aucs = {}
    plt.figure(figsize=figsize)
    
    #Get marginals
    model_pr_rates = {
        model:precision_recall_curve(true_labels, marginals_df[model])
        for model in marginals_df.columns
    }
    model_aucs = {
        model:auc(model_pr_rates[model][1], model_pr_rates[model][0]) 
        for model in model_pr_rates
    }
    if model_type == 'barplot':
        display_df = (
            pd.DataFrame
            .from_dict(model_aucs,orient='index', columns=["auc"])
            .reset_index()
            .rename(index=str, columns={"index": "model"})
        )
        sns.barplot(x='auc', y='model', data=display_df, color='blue')
        plt.xlim(xlim)
        plt.title(plot_title)
    elif model_type == 'curve':
        positive_class = true_labels.sum()/len(true_labels)
        plt.plot([0,1], [positive_class, positive_class], color='grey', 
                 linestyle='--', label='Baseline (AUC = {:0.2f})'.format(positive_class))

        for model in model_pr_rates:
            plt.plot(
                model_pr_rates[model][1], 
                model_pr_rates[model][0], 
                label=model+" (AUC = {:0.2f})".format(model_aucs[model])
            )

        plt.title(plot_title)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
    elif model_type == 'scatterplot':
        display_df = (
            pd.DataFrame
            .from_dict(model_aucs,orient='index', columns=["auc"])
            .reset_index()
            .rename(index=str, columns={"index": "model"})
        )
        sns.pointplot(x='model', y='auc', data=display_df)
        plt.title(plot_title)
    else:
        raise Exception("Please pick a valid option: barplot, curve, scatterplot")
    return 


def plot_label_matrix_heatmap(L, plot_title="Label Matrix", figsize=(10,6), colorbar=True, **kwargs):
    """
    This code is "borrowed" from the snorkel metal repository.
    It diplays Label Functions in the form of a heatmap. 
    WIll be usefule to see how label functions are firing on a particualr dataset
    

    L - a matrix that contains either label function output, 
        number of overlaps or number of conflicts.

    return None but plots label function diagnostics
    """

    plt.figure(figsize=figsize)
    L = L.todense() if sparse.issparse(L) else L
    plt.imshow(L, aspect="auto")
    plt.title(plot_title)

    if "xaxis_tick_labels" in kwargs:
        xtick_pos = range(len(kwargs["xaxis_tick_labels"]))
        plt.xticks(xtick_pos, kwargs["xaxis_tick_labels"], rotation='vertical')


    if "yaxis_tick_labels" in kwargs:
        ytick_pos = range(len(kwargs["yaxis_tick_labels"]))
        plt.yticks(ytick_pos, kwargs["yaxis_tick_labels"])

    if colorbar:
        labels = sorted(np.unique(np.asarray(L).reshape(-1, 1).squeeze()))
        boundaries = np.array(labels + [max(labels) + 1]) - 0.5
        plt.colorbar(boundaries=boundaries, ticks=labels)

def plot_generative_model_weights(gen_model, lf_names, plot_title="Gen Model Weights", figsize=(10,6)):
    plt.figure(figsize=figsize)
    lf_df = pd.DataFrame(gen_model.weights.lf_accuracy.T, columns=["weights"])
    lf_df['label_functions'] = lf_names
    sns.barplot(x="weights", y="label_functions", data=lf_df)
    plt.title(plot_title)
    return