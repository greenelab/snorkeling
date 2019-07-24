import os
import glob
from collections import OrderedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import re

plt.switch_backend('agg')


def get_dataframes(
    result_dir, file_path,
    starting_point=0,
    ending_point=30,
    step=5, num_of_points=4
):
    """
     This function grabs the result tsv files
     and loads then into a dictionary strucutre
     [relationship] -> dataframe

     Args:
         result_dir - the directory containing all the results
         file_path - the path to extract the result files
         starting_point - the point to start each subgraph with in plot_graph function
         ending_point - the point to end each subgraph with
         step - the number to increase the middle points with
         num_of_points - the number of points to plot between the start and end points
    """
    query_points = [0.25 * i for i in range(5)]

    return {
        # Get the head word of each file that will be parsed
        re.search("(dev|test)", file).group(0):
        pd.read_csv(file, sep="\t")
        .assign(num_lfs=lambda x: x['num_lfs'].map(lambda y: y if y != 'baseline' else 'BL'))
        .query("num_lfs in @query_points", engine="python", local_dict={"query_points":query_points})
        for file in glob.glob(f"{result_dir}{file_path}")
    }


def plot_performance_graph(
    metric='AUROC',
    evaluation_set='dev',
    title="",
    file_name="",
    data=None,
    color_map=None
):
    """
    Plot the graphs onto a multi-subplot grid using seaborn
    Args:
        metric - the metric to plot for the y axis
        evaluation_set - whehter to plot the dev set or test set
        title - the main title of the large graph
        file_name - the name of the file to save the graph
        data - the dataframe tree to plot the large graph
        color_map - the color coded to plot each point on
    """
    fig, axes = plt.subplots(len(file_tree["DaG"]), len(file_tree),  figsize=(25, 15), sharey='row')

    for row_ind, col in enumerate(data):

        for col_ind, row in enumerate(data[col]):

            perform_df = data[col][row][evaluation_set].copy()
            perform_df['num_lfs'] = perform_df['num_lfs'] * 100

            # plot the graph
            sns.pointplot(
                x="num_lfs", y=metric,
                data=perform_df.astype({"num_lfs": int}),
                ax=axes[col_ind][row_ind], ci="sd",
                scale=1.25
            )

            # remove x axis labels
            axes[col_ind][row_ind].set_xlabel('')

            if metric == "AUROC":
                axes[col_ind][row_ind].set_ylim([0, 1])

            if metric == "AUPRC":
                axes[col_ind][row_ind].set_ylim([0, 0.5])

            # only set first column and first row titles
            if col_ind == 0:
                axes[col_ind][row_ind].set_title(col, color=color_map[col])

            if row_ind == 0:
                axes[col_ind][row_ind].set_ylabel(row, fontsize=30)
            else:
                axes[col_ind][row_ind].set_ylabel('')

    # Change the font for each element of the graph
    for item in axes.flat:
        item.title.set_fontsize(30)
        item.yaxis.label.set_fontsize(25)
        item.xaxis.label.set_fontsize(25)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(23)

    # Add the subtitles and save the graph
    fig.text(0.04, 0.5, 'Label Function Polarity', va='center', rotation='vertical', fontsize=26)
    fig.text(0.5, 0.04, 'Frequency of Emissions (%)', ha='center', fontsize=30)
    fig.text(0.5, 0.90, f'Predicted Relations ({metric})', ha='center',  fontsize=25)
    fig.suptitle(title, fontsize=30)
    #fig.text(0.69, 0.02, '0-Only Uses Relation Specific Databases.', fontsize=27)
    plt.subplots_adjust(top=0.85)
    plt.savefig(file_name, format='png')

# If running the script itself
# execute the code below
if __name__ == '__main__':
    file_tree = OrderedDict({
        "DaG":
        {
            "pos": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/error_analysis/pos",
            "neg": "../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/error_analysis/neg",

        },
        "CtD":
        {
            "pos": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/error_analysis/pos",
            "neg": "../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/error_analysis/neg",

        },

        "CbG":
        {
            "pos": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/error_analysis/pos",
            "neg": "../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/error_analysis/neg",

        },
        "GiG":
        {
            "pos": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/error_analysis/pos",
            "neg": "../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/error_analysis/neg",

        }
    })


    # Use the file tree above and graph the appropiate files
    performance_data_tree = OrderedDict({
        key: {
            sub_key: get_dataframes(
                file_tree[key][sub_key], "*performance.tsv",
            )
            for sub_key in file_tree[key]
        }
        for key in file_tree
    })

    # Obtained from color brewer 2
    # 5 classes using the supposedly color blind friendly colors
    color_names = {
        "turquoise": pd.np.array([27, 158, 119])/255,
        "orange": pd.np.array([217, 95, 2])/255,
        "purple": pd.np.array([117, 112, 179])/255,
        "pink": pd.np.array([231, 41, 138])/255,
        "light-green": pd.np.array([102, 166, 30])/255
    }

    color_map = {
        "DaG": color_names["turquoise"],
        "CtD": color_names["orange"],
        "CbG": color_names["purple"],
        "GiG": color_names["pink"],
        "All": color_names["light-green"]
    }

    plot_performance_graph(
        metric="AUROC", evaluation_set='dev',
        title="Random Label Function Generative Model Assessment (Dev Set)",
        file_name="../transfer_dev_set_auroc.png", data=performance_data_tree,
        color_map=color_map
    )
    plot_performance_graph(
        metric="AUPRC",  evaluation_set='dev',
        title="Random Label Function Generative Model Assessment (Dev Set)",
        file_name="../transfer_dev_set_auprc.png", data=performance_data_tree,
        color_map=color_map
    )
    plot_performance_graph(
        metric="AUROC", evaluation_set='test',
        title="Random Label Function Generative Model Assessment (Test Set)",
        file_name="../transfer_test_set_auroc.png", data=performance_data_tree,
        color_map=color_map
    )
    plot_performance_graph(
        metric="AUPRC", evaluation_set='test',
        title="Random Label Function Generative Model Assessment (Test Set)",
        file_name="../transfer_test_set_auprc.png", data=performance_data_tree,
        color_map=color_map
    )
