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
        re.search("(precision)", file).group(0):
        (
            pd.read_csv(file, sep="\t")
            .round({"precision": 2})
            .drop_duplicates(["precision", "in_hetionet"])
            .sort_values("precision", ascending=False)
        )
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
    fig, axes = plt.subplots(len(file_tree["DaG"]), len(file_tree), figsize=(22, 10), sharey='row', sharex='row')

    for row_ind, col in enumerate(data):
        for row in data[col]:

            if "precision" not in data[col][row]:
                continue
            data[col][row][evaluation_set] = (
                data[col][row][evaluation_set]
                .groupby([
                    pd.cut(
                        data[col][row][evaluation_set]["precision"],
                        pd.np.arange(0, 1, 0.1)
                    ),
                    "in_hetionet"
                ])
                .max()
                .reset_index(level="in_hetionet")
                .reset_index(drop=True)
                .append(data[col][row][evaluation_set].query("precision==1"), sort=True)
            )

            # plot the graph
            sns.scatterplot(
                x="precision", y=metric,
                hue="in_hetionet",
                data=data[col][row][evaluation_set].sort_values("in_hetionet"),
                ax=axes[row_ind], ci=None,
            )

            # print(data[col][row][evaluation_set])

            # remove x axis labels
            axes[row_ind].set_xlabel('')

            axes[row_ind].set_title(row, color=color_map[row])
            axes[row_ind].set(yscale="log")
            axes[row_ind].set_ylim([10**(-1/8), 10e4])
            axes[row_ind].set_xlim([0, 1.05])
            axes[row_ind].set_ylabel("")
            axes[row_ind].get_legend().remove()

    # Change the font for each element of the graph
    for item in axes.flat:
        item.title.set_fontsize(20)
        item.yaxis.label.set_fontsize(15)
        item.xaxis.label.set_fontsize(15)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(15)

    axes.flatten()[2].legend(loc='upper center', bbox_to_anchor=(2.6, 1.0), fontsize=15)
    # Add the subtitles and save the graph
    #fig.text(0.5, 0.89, '', ha='center', fontsize=26)
    fig.text(0.5, 0.02, 'Precision Level', ha='center', fontsize=20)
    fig.text(0.04, 0.5, f'Number of Edges', va='center', rotation='vertical', fontsize=20)
    fig.suptitle(title, fontsize=25)
    plt.subplots_adjust(top=0.85, right=0.85)
    plt.savefig(file_name, format='png')

# If running the script itself
# execute the code below
if __name__ == '__main__':
    file_tree = OrderedDict({
        "DaG":
        {
            "DaG": "../../../disease_gene/disease_associates_gene/edge_prediction_experiment/results/"
        },
        "CtD":
        {
            "CtD": "../../../compound_disease/compound_treats_disease/edge_prediction_experiment/results/"
        },

        "CbG":
        {
            "CbG": "../../../compound_gene/compound_binds_gene/edge_prediction_experiment/results/"

        },
        "GiG":
        {
            "GiG": "../../../gene_gene/gene_interacts_gene/edge_prediction_experiment/results/"

        }
    })


    # Use the file tree above and graph the appropiate files
    edges_data_tree = OrderedDict({
        key: {
            sub_key: get_dataframes(
                file_tree[key][sub_key], "*edges_added.tsv",
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
        metric="edges", evaluation_set='precision',
        title="Reconstructing Hetionet",
        file_name="../edges_added.png", data=edges_data_tree,
        color_map=color_map
    )
