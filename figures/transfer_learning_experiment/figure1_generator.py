import os
import glob
from collections import OrderedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_dataframes(result_dir):
    return {
        # Get the head word of each file that will be parsed
        os.path.splitext(os.path.basename(file))[0].split("_")[0]:
        pd.read_csv(file, sep="\t")
        .assign(num_lfs=lambda x: x['num_lfs'].map(lambda y: y if y != 'baseline' else 'BL'))
        for file in glob.glob(f"{result_dir}/*.tsv")
    }


def plot_performance_graph(metric='AUROC', evaluation_set='dev', title="", file_name=""):
    fig, axes = plt.subplots(len(file_tree), len(file_tree["DaG"]), figsize=(25, 15), sharey='row')

    for row_ind, col in enumerate(data_tree):
        for col_ind, row in enumerate(data_tree[col]):

            sns.pointplot(x="num_lfs", y=metric, data=data_tree[col][row][evaluation_set], ax=axes[row_ind][col_ind])
            axes[row_ind][col_ind].set_xlabel('')

            if row == "All":
                labels = sorted(axes[row_ind][col_ind].get_xticklabels(), key=lambda x: int(x.get_text()))

                for i, l in enumerate(labels):
                    if l not in labels[0::3] + labels[-1:]:
                        labels[i] = ''

                axes[row_ind][col_ind].set_xticklabels(labels)

            if row_ind == 0:
                axes[row_ind][col_ind].set_title(row)

            if col_ind == 0:
                axes[row_ind][col_ind].set_ylabel(col)
            else:
                axes[row_ind][col_ind].set_ylabel('')

    for item in axes.flat:
        item.title.set_fontsize(17.5)
        item.yaxis.label.set_fontsize(19)
        item.xaxis.label.set_fontsize(15)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(14)

    fig.text(0.5, 0.04, 'Number of Additional Label Functions', ha='center')
    fig.text(0.04, 0.5, f'Predicted Relations ({metric})', va='center', rotation='vertical')
    fig.suptitle(title)
    fig.text(0.7, 0.02, '0-Only uses relation specific databases. (Distant Supervision)', fontsize=14)
    plt.savefig(file_name, format='png')


file_tree = OrderedDict({
    "DaG":
    {
        "Disease associates Gene (DaG)": "../../disease_gene/disease_associates_gene/single_task/data/random_sampling/DaG/results",
        "Compound treats Disease (CtD)": "../../disease_gene/disease_associates_gene/single_task/data/random_sampling/CtD/results",
        "Compound binds Gene (CbG)": "../../disease_gene/disease_associates_gene/single_task/data/random_sampling/CbG/results",
        "Gene interacts Gene (GiG)": "../../disease_gene/disease_associates_gene/single_task/data/random_sampling/GiG/results",
        "All": "../../disease_gene/disease_associates_gene/single_task/data/random_sampling/all",
    },
    "CtD":
    {
        "Disease associates Gene (DaG)": "../../compound_disease/compound_treats_disease/single_task/data/random_sampling/DaG/results",
        "Compound treats Disease (CtD)": "../../compound_disease/compound_treats_disease/single_task/data/random_sampling/CtD/results",
        "Compound binds Gene (CbG)": "../../compound_disease/compound_treats_disease/single_task/data/random_sampling/CbG/results",
        "Gene interacts Gene (GiG)": "../../compound_disease/compound_treats_disease/single_task/data/random_sampling/GiG/results",
        "All": "../../compound_disease/compound_treats_disease/single_task/data/random_sampling/all",
    },

    "CbG":
    {
        "Disease associates Gene (DaG)": "../../compound_gene/compound_binds_gene/single_task/data/random_sampling/DaG/results",
        "Compound treats Disease (CtD)": "../../compound_gene/compound_binds_gene/single_task/data/random_sampling/CtD/results",
        "Compound binds Gene (CbG)": "../../compound_gene/compound_binds_gene/single_task/data/random_sampling/CbG/results",
        "Gene interacts Gene (GiG)": "../../compound_gene/compound_binds_gene/single_task/data/random_sampling/GiG/results",
        "All": "../../compound_gene/compound_binds_gene/single_task/data/random_sampling/all",
    },
    "GiG":
    {
        "Disease associates Gene (DaG)": "../../gene_gene/gene_interacts_gene/single_task/data/random_sampling/DaG/results",
        "Compound treats Disease (CtD)": "../../gene_gene/gene_interacts_gene/single_task/data/random_sampling/CtD/results",
        "Compound binds Gene (CbG)": "../../gene_gene/gene_interacts_gene/single_task/data/random_sampling/CbG/results",
        "Gene interacts Gene (GiG)": "../../gene_gene/gene_interacts_gene/single_task/data/random_sampling/GiG/results",
        "All": "../../gene_gene/gene_interacts_gene/single_task/data/random_sampling/all",
    }
})


data_tree = OrderedDict({
    key: {
        sub_key: get_dataframes(file_tree[key][sub_key])
        for sub_key in file_tree[key]
    }
    for key in file_tree
})

plt.rcParams.update({'font.size': 22})

plot_performance_graph(
    metric="AUROC", evaluation_set='dev',
    title="Stepwise Label Function Assessment (Development Set)",
    file_name="transfer_dev_set_auroc.png"
)
plot_performance_graph(
    metric="AUPRC",  evaluation_set='dev',
    title="Stepwise Label Function Assessment (Development Set)",
    file_name="transfer_dev_set_auprc.png"
)
plot_performance_graph(
    metric="AUROC", evaluation_set='test',
    title="Stepwise Label Function Assessment (Test Set)",
    file_name="transfer_test_set_auroc.png"
)
plot_performance_graph(
    metric="AUPRC", evaluation_set='test',
    title="Stepwise Label Function Assessment (Test Set)",
    file_name="transfer_test_set_auprc.png"
)
