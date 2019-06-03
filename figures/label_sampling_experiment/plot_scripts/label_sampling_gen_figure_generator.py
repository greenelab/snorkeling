import os
import glob
from collections import OrderedDict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

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
    # Build up X axis by gathering relatively evely spaced points
    query_points = [starting_point]
    query_points += [1 + step*index for index in range(num_of_points)]
    query_points += [ending_point]

    return {
        # Get the head word of each file that will be parsed
        os.path.splitext(os.path.basename(file))[0].split("_")[0]:
        pd.read_csv(file, sep="\t")
        .assign(num_lfs=lambda x: x['num_lfs'].map(lambda y: y if y != 'baseline' else 'BL'))
        .query("num_lfs in @query_points", engine="python", local_dict={"query_points":query_points})
        for file in glob.glob(f"{result_dir}/{file_path}")
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
    fig, axes = plt.subplots(len(file_tree), len(file_tree["DaG"]), figsize=(25, 15), sharey='row')

    for row_ind, col in enumerate(data):
        for col_ind, row in enumerate(data[col]):
            # plot the graph
            sns.pointplot(
                x="num_lfs", y=metric,
                data=data[col][row][evaluation_set],
                ax=axes[row_ind][col_ind], ci="sd",
                scale=1.25
            )

            # remove x axis labels
            axes[row_ind][col_ind].set_xlabel('')

            # Not used now because of bad asethetics
            # Set Y axis to have 0.5
            #if metric=="AUROC":
            #    axes[row_ind][col_ind].set_ylim(
            #        bottom=0.5,
            #        top=pd.np.ceil(data[col][row][evaluation_set][metric].max()*10)/10 - 0.02
            #    )

            # unstable code
            # if order of graph elements
            # change then this code will not work
            for idx, item in enumerate(axes[row_ind][col_ind].get_children()):

                # if the points in graph
                # change color map accordingly
                if idx == 0:
                    item.set_edgecolor([
                        color_map[col] if index==0 else color_map[row] 
                        for index in range(len(data[col][row][evaluation_set].num_lfs.unique()))
                    ])
                    item.set_facecolor([
                        color_map[col] if index==0 else color_map[row] 
                        for index in range(len(data[col][row][evaluation_set].num_lfs.unique()))
                    ])

                #if error bars change accordingly
                elif isinstance(item, plt.Line2D):
                    if idx == 1:
                        #item.set_linestyle('dashed')
                        item.set_color("black")

                    else:
                        item.set_color(color_map[row])

            # only set first column and first row titles
            if row_ind == 0:
                axes[row_ind][col_ind].set_title(row, color=color_map[row])

            if col_ind == 0:
                axes[row_ind][col_ind].set_ylabel(col, color=color_map[col])
            else:
                axes[row_ind][col_ind].set_ylabel('')

    # Change the font for each element of the graph
    for item in axes.flat:
        item.title.set_fontsize(30)
        item.yaxis.label.set_fontsize(25)
        item.xaxis.label.set_fontsize(25)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(23)

    # Add the subtitles and save the graph
    fig.text(0.5, 0.89, 'Label Function Sources', ha='center', fontsize=26)
    fig.text(0.5, 0.04, 'Number of Additional Label Functions', ha='center', fontsize=30)
    fig.text(0.04, 0.5, f'Predicted Relations ({metric})', va='center', rotation='vertical', fontsize=25)
    fig.suptitle(title, fontsize=30)
    fig.text(0.69, 0.02, '0-Only Uses Relation Specific Databases.', fontsize=27)
    plt.subplots_adjust(top=0.85)
    plt.savefig(file_name, format='png')

# If running the script itself
# execute the code below
if __name__ == '__main__':
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

    # End total of label functions for each point
    end_points = {
        "DaG": 30,
        "CtD": 22,
        "CbG": 20,
        "GiG": 28,
        "All": 100
    }

    # Use the file tree above and graph the appropiate files
    performance_data_tree = OrderedDict({
        key: {
            sub_key: get_dataframes(
                file_tree[key][sub_key], "*sampled_performance.tsv",
                ending_point=end_points[sub_key],
                # if using all the label functions step by 32 instead of 5
                step=5 if sub_key !="All" else 32
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
        "DaG": color_names["turquoise"]
        "CtD": color_names["orange"]
        "CbG": color_names["purple"]
        "GiG": color_names["pink"]
        "All": color_names["light-green"]
    }

    plot_performance_graph(
        metric="AUROC", evaluation_set='dev',
        title="Label Sampling Generative Model Assessment (Dev Set)",
        file_name="../transfer_dev_set_auroc.png", data=performance_data_tree,
        color_map=color_map
    )
    plot_performance_graph(
        metric="AUPRC",  evaluation_set='dev',
        title="Label Sampling Generative Model Assessment (Dev Set)",
        file_name="../transfer_dev_set_auprc.png", data=performance_data_tree,
        color_map=color_map
    )
    plot_performance_graph(
        metric="AUROC", evaluation_set='test',
        title="Label Sampling Generative Model Assessment (Test Set)",
        file_name="../transfer_test_set_auroc.png", data=performance_data_tree,
        color_map=color_map
    )
    plot_performance_graph(
        metric="AUPRC", evaluation_set='test',
        title="Label Sampling Generative Model Assessment (Test Set)",
        file_name="../transfer_test_set_auprc.png", data=performance_data_tree,
        color_map=color_map
    )
