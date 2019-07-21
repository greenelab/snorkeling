import os
import glob
from collections import OrderedDict
from sklearn.calibration import calibration_curve
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
    step=5, num_of_points=4,
    model_criteria=None
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
    if "xlsx" in file_path:
        return{
            "labels":
            (
                pd.read_excel(file, sep="\t")
                .query(f"{model_criteria}.notnull()")
                .sort_values("candidate_id")
                [model_criteria]
            )
            for file in glob.glob(f"{result_dir}{file_path}")
        }

    return {
        # Get the head word of each file that will be parsed
        re.search("(before|after)", file).group(0):
        pd.read_csv(file, sep="\t")
        for file in glob.glob(f"{result_dir}{file_path}")
    }


def plot_performance_graph(
    metric='AUROC',
    evaluation_labels=None,
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
    fig, axes = plt.subplots(len(file_tree["DaG"]), len(file_tree), figsize=(18, 8), sharey='row', sharex='row')
    for row_ind, col in enumerate(data):
        for row in data[col]:

            # plot the graph
            prob_true_before, prob_pred_before = calibration_curve(
                evaluation_labels[col][row]["labels"].values,
                data[col][row]['before'][metric].values
            )
            prob_true_after, prob_pred_after = calibration_curve(
                evaluation_labels[col][row]["labels"].values,
                data[col][row]['after'][metric].values
            )

            axes[row_ind].plot(
                prob_pred_before,
                prob_true_before,
                marker='o',
                label="Before Calibration"
            )

            axes[row_ind].plot(
                prob_pred_after,
                prob_true_after,
                marker='o',
                label="After Calibration"
            )

            axes[row_ind].plot(
                [0, 1], [0, 1],
                color='black',
                linestyle='--',
                label="Perfectly Calibrated"
            )

            # remove x axis labels
            axes[row_ind].set_xlabel('')
            axes[row_ind].set_title(row, color=color_map[row])
            axes[row_ind].set_ylabel("")

    # Change the font for each element of the graph
    for item in axes.flat:
        item.title.set_fontsize(20)
        item.yaxis.label.set_fontsize(15)
        item.xaxis.label.set_fontsize(15)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(15)

    axes.flatten()[3].legend(loc='upper center', bbox_to_anchor=(1.48, 1.0), fontsize=13)
    # Add the subtitles and save the graph
    #fig.text(0.5, 0.89, '', ha='center', fontsize=26)
    fig.text(0.48, 0.02, 'Predicted', ha='center', fontsize=20)
    fig.text(0.06, 0.5, f'Actual', va='center', rotation='vertical', fontsize=20)
    fig.suptitle(title, fontsize=30)
    plt.subplots_adjust(top=0.85, right=.83)
    plt.savefig(file_name, format='png')

# If running the script itself
# execute the code below
if __name__ == '__main__':
    file_tree = OrderedDict({
        "DaG":
        {
            "DaG": "../../../disease_gene/disease_associates_gene/model_calibration_experiment/results/"
        },
        "CtD":
        {
            "CtD": "../../../compound_disease/compound_treats_disease/model_calibration_experiment/results/"
        },

        "CbG":
        {
            "CbG": "../../../compound_gene/compound_binds_gene/model_calibration_experiment/results/"

        },
        "GiG":
        {
            "GiG": "../../../gene_gene/gene_interacts_gene/model_calibration_experiment/results/"

        }
    })


    # Use the file tree above and graph the appropiate files
    calibration_data_tree = OrderedDict({
        key: {
            sub_key: get_dataframes(
                file_tree[key][sub_key], "*dev.tsv",
            )
            for sub_key in file_tree[key]
        }
        for key in file_tree
    })

    sen_file_tree = OrderedDict({
        "DaG":
        {
            "DaG": "../../../disease_gene/disease_associates_gene/data/sentences/",
            "model_criteria": "curated_dsh"
        },
        "CtD":
        {
            "CtD": "../../../compound_disease/compound_treats_disease/data/sentences/",
            "model_criteria": "curated_ctd"
        },

        "CbG":
        {
            "CbG": "../../../compound_gene/compound_binds_gene/data/sentences/",
            "model_criteria": "curated_cbg"

        },
        "GiG":
        {
            "GiG": "../../../gene_gene/gene_interacts_gene/data/sentences/",
            "model_criteria": "curated_gig"

        }
    })

    # Use the file tree above and graph the appropiate files
    sentences_data_tree = OrderedDict({
        key: {
            sub_key: get_dataframes(
                sen_file_tree[key][sub_key], "*dev.xlsx",
                model_criteria=sen_file_tree[key]["model_criteria"]
            )
            for sub_key in sen_file_tree[key] if sub_key != "model_criteria"
        }
        for key in sen_file_tree
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
        metric="model_prediction", evaluation_labels=sentences_data_tree,
        title="Calibrating Discriminator Model",
        file_name="../model_calibration.png", data=calibration_data_tree,
        color_map=color_map
    )
