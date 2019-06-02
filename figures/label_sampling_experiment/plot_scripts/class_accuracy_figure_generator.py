import os
import glob
from collections import OrderedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

plt.switch_backend('agg')

def get_dataframes(result_dir, file_path):
    return {
        # Get the head word of each file that will be parsed
        os.path.splitext(os.path.basename(file))[0].split("_")[0]:
        pd.read_csv(file, sep="\t")
        .assign(num_lfs=lambda x: x['num_lfs'].map(lambda y: y if y != 'baseline' else 'BL'))
        .query("num_lfs in [0,1,6,11,16,20,28,22,30] or num_lfs in [0,1,33,65,97,100]")
        for file in glob.glob(f"{result_dir}/{file_path}")
    }


def plot_performance_graph(metric='AUROC', evaluation_set='dev', title="", file_name="", data=None):
    fig, axes = plt.subplots(len(file_tree), len(file_tree["DaG"]), figsize=(25, 15), sharey='row')

    for row_ind, col in enumerate(data):
        
        for col_ind, row in enumerate(data[col]):
            sns.pointplot(x="num_lfs", y=metric, data=data[col][row][evaluation_set], ax=axes[row_ind][col_ind])
            axes[row_ind][col_ind].set_xlabel('')

            if row == "All" and len(data[col][row]) != 0:
                labels = sorted(axes[row_ind][col_ind].get_xticklabels(), key=lambda x: int(x.get_text()))
                axes[row_ind][col_ind].set_xticklabels(labels)

            if row_ind == 0:
                axes[row_ind][col_ind].set_title(row)

            if col_ind == 0:
                axes[row_ind][col_ind].set_ylabel(col)
            else:
                axes[row_ind][col_ind].set_ylabel('')


    for item in axes.flat:
        item.title.set_fontsize(30)
        item.yaxis.label.set_fontsize(25)
        item.xaxis.label.set_fontsize(25)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(23)

    fig.text(0.5, 0.04, 'Number of Additional Label Functions', ha='center', fontsize=30)
    fig.text(0.04, 0.5, f'Predicted Relations ({metric})', va='center', rotation='vertical', fontsize=25)
    fig.suptitle(title, fontsize=30)
    fig.text(0.69, 0.02, '0-Only Uses Relation Specific Databases.', fontsize=27)
    plt.savefig(file_name, format='png')


file_tree = OrderedDict({
    "DaG":
    {
        "DaG": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/DaG/results",
        "CtD": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/CtD/results",
        "CbG": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/CbG/results",
        "GiG": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/GiG/results",
        "All": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/all",
    },
    "CtD":
    {
        "DaG": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/DaG/results",
        "CtD": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/CtD/results",
        "CbG": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/CbG/results",
        "GiG": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/GiG/results",
        "All": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/all",
    },

    "CbG":
    {
        "DaG": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/DaG/results",
        "CtD": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/CtD/results",
        "CbG": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/CbG/results",
        "GiG": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/GiG/results",
        "All": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/all",
    },
    "GiG":
    {
        "DaG": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/DaG/results",
        "CtD": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/CtD/results",
        "CbG": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/CbG/results",
        "GiG": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/GiG/results",
        "All": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/all",
    }
})

performance_data_tree = OrderedDict({
    key: {
        sub_key: get_dataframes(file_tree[key][sub_key], "*sampled_performance.tsv")
        for sub_key in file_tree[key]
    }
    for key in file_tree
})

plt.rcParams.update({'font.size': 22})

class_correct_data_tree = OrderedDict({
    key: {
        sub_key: get_dataframes(file_tree[key][sub_key], "*marginals.tsv")
        for sub_key in file_tree[key]
    }
    for key in file_tree
})


plot_performance_graph(
    metric="frac_correct", evaluation_set='dev',
    title="Individual Class Performance (Dev Set)",
    file_name="class_correct_dev_set.png", data=class_correct_data_tree
)

plot_performance_graph(
    metric="frac_correct", evaluation_set='test',
    title="Individual Class Performance (Test Set)",
    file_name="class_correct_test_set.png", data=class_correct_data_tree
)
