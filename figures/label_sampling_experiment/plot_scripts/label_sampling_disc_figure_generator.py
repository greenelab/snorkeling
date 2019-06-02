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


def plot_performance_graph(metric='AUROC', evaluation_set='dev', title="", file_name="", data=None, color_map=None):
    fig, axes = plt.subplots(len(file_tree), len(file_tree["DaG"]), figsize=(25, 15), sharey='row')

    for row_ind, col in enumerate(data):
        for col_ind, row in enumerate(data[col]):

            # Data Not Available Yet
            if len(data[col][row]) == 0:
                lower, upper = axes[row_ind][col_ind].get_ylim()
                axes[row_ind][col_ind].annotate("Coming Soon!!", (0.2, (lower+upper)/2), color="red", fontsize=20)
                
            else:
                sns.pointplot(
                    x="num_lfs", y=metric, 
                    data=data[col][row][evaluation_set], 
                    ax=axes[row_ind][col_ind], 
                    hue="label", ci="sd", scale=1.2,
                    markers=["^", "o"]
                )

                # remove x axis labels
                axes[row_ind][col_ind].set_xlabel('')
                axes[row_ind][col_ind].get_legend().remove()

                # unstable code
                # if order of error bars
                # change then this code will not work
                for idx, item in enumerate(axes[row_ind][col_ind].get_children()):
                    # if the points in graph 
                    # change color map accordingly
                    if idx == 0 or idx==1:
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
                        if idx == 2:
                            item.set_linestyle('dashed')
                            item.set_color("black")
                            item.set_alpha(0.25)
                        elif idx == 9:
                            item.set_linestyle('dashed')
                            item.set_color("black")
                            item.set_alpha(0.25)
                        else:
                            item.set_color(color_map[row])
                
            # only set first column and first row titles
            if row_ind == 0:
                axes[row_ind][col_ind].set_title(row, color=color_map[row])

            if col_ind == 0:
                axes[row_ind][col_ind].set_ylabel(col, color=color_map[col])
            else:
                axes[row_ind][col_ind].set_ylabel('')

    for item in axes.flat:
        item.title.set_fontsize(30)
        item.yaxis.label.set_fontsize(24)
        item.xaxis.label.set_fontsize(24)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(23)

    if "label" in data["DaG"]["DaG"]["dev"].columns:
        axes.flatten()[3].legend(loc='upper center', bbox_to_anchor=(2.53, 0.8), fontsize=20)
        leg = axes.flatten()[3].get_legend()
        leg.legendHandles[0].set_edgecolor('black')
        leg.legendHandles[0].set_facecolor('white')
        
        leg.legendHandles[1].set_edgecolor('black')
        leg.legendHandles[1].set_facecolor('white')
    
    fig.text(0.5, 0.89, 'Label Sources', ha='center', fontsize=30)
    fig.text(0.5, 0.04, 'Number of Additional Label Functions', ha='center', fontsize=30)
    fig.text(0.04, 0.5, f'Predicted Relations ({metric})', va='center', rotation='vertical', fontsize=25)
    fig.suptitle(title, fontsize=30)
    fig.text(0.69, 0.02, '0-Only Uses Relation Specific Databases.', fontsize=27)
    plt.subplots_adjust(top=0.85)
    plt.savefig(file_name, format='png')


color_map = {
    "DaG": pd.np.array([27, 158, 119])/255,
    "CtD": pd.np.array([217, 95, 2])/255,
    "CbG": pd.np.array([117, 112, 179])/255,
    "GiG": pd.np.array([231, 41, 138])/255,
    "All": pd.np.array([102, 166, 30])/255
}

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

disc_performance_tree = OrderedDict({
    key: {
        sub_key: get_dataframes(file_tree[key][sub_key], "*disc_performance.tsv")
        for sub_key in file_tree[key]
    }
    for key in file_tree
})

plot_performance_graph(
    metric="AUROC", evaluation_set='dev',
    title="Label Sampling Discriminator Model Assessment (Dev Set)",
    file_name="../disc_performance_dev_set_auroc.png", data=disc_performance_tree,
    color_map=color_map
)

plot_performance_graph(
    metric="AUPRC", evaluation_set='dev',
    title="Label Sampling Discriminator Model Assessment (Dev Set)",
    file_name="../disc_performance_dev_set_auprc.png", data=disc_performance_tree,
    color_map=color_map
)

plot_performance_graph(
    metric="AUROC", evaluation_set='test',
    title="Label Sampling Discriminator Model Assessment (Test Set)",
    file_name="../disc_performance_test_set_auroc.png", data=disc_performance_tree,
    color_map=color_map
)

plot_performance_graph(
    metric="AUPRC", evaluation_set='test',
    title="Label Sampling Discriminator Model Assessment (Test Set)",
    file_name="../disc_performance_test_set_auprc.png", data=disc_performance_tree,
    color_map=color_map
)
