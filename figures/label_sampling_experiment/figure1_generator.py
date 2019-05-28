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
        for file in glob.glob(f"{result_dir}/{file_path}")
    }


def plot_performance_graph(metric='AUROC', evaluation_set='dev', title="", file_name="", data=None):
    fig, axes = plt.subplots(len(file_tree), len(file_tree["DaG"]), figsize=(25, 15), sharey='row')

    for row_ind, col in enumerate(data):
        for col_ind, row in enumerate(data[col]):

            if len(data[col][row]) == 0:
                lower, upper = axes[row_ind][col_ind].get_ylim()
                axes[row_ind][col_ind].annotate("Coming Soon!!", (0.2, (lower+upper)/2), color="red", fontsize=20)

            elif metric == "frac_correct":
                data[col][row][evaluation_set].label.replace([0, 1], ["negative", "positive"], inplace=True)
                sns.pointplot(
                    x="num_lfs", y=metric,
                    data=data[col][row][evaluation_set],
                    ax=axes[row_ind][col_ind],
                    hue="label"
                )
                axes[row_ind][col_ind].get_legend().remove()
            else:
                if "label" in data[col][row][evaluation_set].columns:
                    sns.pointplot(x="num_lfs", y=metric, data=data[col][row][evaluation_set], ax=axes[row_ind][col_ind], hue="label")
                    axes[row_ind][col_ind].get_legend().remove()
                else:
                    sns.pointplot(x="num_lfs", y=metric, data=data[col][row][evaluation_set], ax=axes[row_ind][col_ind])

            axes[row_ind][col_ind].set_xlabel('')

            if row == "All" and len(data[col][row]) != 0:
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
        item.title.set_fontsize(18)
        item.yaxis.label.set_fontsize(24)
        item.xaxis.label.set_fontsize(24)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(14)

    if metric == "frac_correct":
        axes.flatten()[4].legend(loc='upper center', bbox_to_anchor=(1.3, 0.7), fontsize=20)
        metric = "#correct / total"

    elif "label" in data["DaG"]["Disease associates Gene (DaG)"]["dev"].columns:
        axes.flatten()[3].legend(loc='upper center', bbox_to_anchor=(2.5, 0.7), fontsize=20)

    fig.text(0.5, 0.04, 'Number of Additional Label Functions', ha='center', fontsize=25)
    fig.text(0.04, 0.5, f'Predicted Relations ({metric})', va='center', rotation='vertical', fontsize=25)
    fig.suptitle(title, fontsize=30)
    fig.text(0.7, 0.02, '0-Only uses relation specific databases.', fontsize=17)
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

performance_data_tree = OrderedDict({
    key: {
        sub_key: get_dataframes(file_tree[key][sub_key], "*sampled_performance.tsv")
        for sub_key in file_tree[key]
    }
    for key in file_tree
})

plt.rcParams.update({'font.size': 22})

plot_performance_graph(
    metric="AUROC", evaluation_set='dev',
    title="Stepwise Label Function Assessment (Devt Set)",
    file_name="transfer_dev_set_auroc.png", data=performance_data_tree
)
plot_performance_graph(
    metric="AUPRC",  evaluation_set='dev',
    title="Stepwise Label Function Assessment (Dev Set)",
    file_name="transfer_dev_set_auprc.png", data=performance_data_tree
)
plot_performance_graph(
    metric="AUROC", evaluation_set='test',
    title="Stepwise Label Function Assessment (Test Set)",
    file_name="transfer_test_set_auroc.png", data=performance_data_tree
)
plot_performance_graph(
    metric="AUPRC", evaluation_set='test',
    title="Stepwise Label Function Assessment (Test Set)",
    file_name="transfer_test_set_auprc.png", data=performance_data_tree
)

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

disc_performance_tree = OrderedDict({
    key: {
        sub_key: get_dataframes(file_tree[key][sub_key], "*disc_performance.tsv")
        for sub_key in file_tree[key]
    }
    for key in file_tree
})

plot_performance_graph(
    metric="AUROC", evaluation_set='dev',
    title="Disc Performance (Dev Set)",
    file_name="disc_performance_dev_set_auroc.png", data=disc_performance_tree
)

plot_performance_graph(
    metric="AUPRC", evaluation_set='dev',
    title="Disc Performance (Dev Set)",
    file_name="disc_performance_dev_set_auprc.png", data=disc_performance_tree
)

plot_performance_graph(
    metric="AUROC", evaluation_set='test',
    title="Disc Performance (Test Set)",
    file_name="disc_performance_test_set_auroc.png", data=disc_performance_tree
)

plot_performance_graph(
    metric="AUPRC", evaluation_set='test',
    title="Disc Performance (Test Set)",
    file_name="disc_performance_test_set_auprc.png", data=disc_performance_tree
)
