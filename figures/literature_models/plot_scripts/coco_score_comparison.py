import os
import glob
from collections import OrderedDict
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
import pdb
import re

plt.switch_backend('agg')


def plot_performance_graph(
    title="",
    file_name="",
    data=None,
    color_map=None
):
    """
    Plot the graphs onto a multi-subplot grid using seaborn
    Args:
        title - the main title of the large graph
        file_name - the name of the file to save the graph
        data - the dataframe tree to plot the large graph
        color_map - the color coded to plot each point on
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex='row')
    
    data_entry = []
    for edge_type in data:

        max_model_df = (
            data[edge_type]["our_model"]
            .groupby(
                ["doid_id", "entrez_gene_id"] if edge_type == "DaG" else 
                ["doid_id", "drugbank_id"] if edge_type =="CtD" else 
                ["drugbank_id", "entrez_gene_id"] if edge_type == "CbG" else
                ["gene1_id", "gene2_id"]
            )
            .agg({
                "model_prediction": 'max', 
                "hetionet": 'max',
            })
        )
        
        fpr, tpr, _ = roc_curve(
            max_model_df["hetionet"], 
            max_model_df["model_prediction"]
        )
        
        precision,recall, _ = precision_recall_curve(
            max_model_df["hetionet"], 
            max_model_df["model_prediction"]
        )
        
        data_entry.append({
            "AUROC":auc(fpr, tpr),
            "AUPR":auc(recall, precision),
            "Models": "Our Model",
            "Edge_Type": edge_type
        })
        
        fpr, tpr, _ = roc_curve(
            data[edge_type]["coco_score"]["hetionet"], 
            data[edge_type]["coco_score"]["final_score"]
        )
        
        precision,recall, _ = precision_recall_curve(
            data[edge_type]["coco_score"]["hetionet"], 
            data[edge_type]["coco_score"]["final_score"]
        )
        
        data_entry.append({
            "AUROC":auc(fpr, tpr),
            "AUPR":auc(recall, precision),
            "Models": "CoCoScore",
            "Edge_Type": edge_type
        })

    performance_df = pd.DataFrame.from_records(data_entry)
    
    print(performance_df)
    
    sns.barplot(x="Edge_Type", y="AUROC", hue="Models", data=performance_df, ax=axes[0])
    sns.barplot(x="Edge_Type", y="AUPR", hue="Models", data=performance_df, ax=axes[1])
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[0].set_ylim([0, 1])
    axes[1].set_ylim([0, 1])
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    
    # Change the font for each element of the graph
    for item in axes.flat:
        item.title.set_fontsize(20)
        item.yaxis.label.set_fontsize(20)
        item.xaxis.label.set_fontsize(20)
        for tick in item.get_yticklabels() + item.get_xticklabels():
            tick.set_fontsize(20)

        for tick in item.get_xticklabels():
            tick.set_color(color_map[tick.get_text()])
            
    axes.flatten()[1].legend(loc='upper center', bbox_to_anchor=(1.2, 1.0), fontsize=15)
    # Add the subtitles and save the graph
    #fig.text(0.5, 0.89, '', ha='center', fontsize=26)
    fig.text(0.5, 0.02, 'Edge Type', ha='center', fontsize=20)
    #fig.text(0.04, 0.5, 'Performance Metric', va='center', rotation='vertical', fontsize=20)
    fig.suptitle(title, fontsize=25)
    plt.subplots_adjust(top=0.85, wspace=0.23, right=0.85)
    plt.savefig(file_name, format='png')

# If running the script itself
# execute the code below
if __name__ == '__main__':
    file_tree = OrderedDict({
        "DaG":
        {
            "coco_score":"../../../disease_gene/disease_associates_gene/literature_models/coco_score/results/dg_edge_prediction_cocoscore.tsv",
            "our_model": "../../../disease_gene/disease_associates_gene/edge_prediction_experiment/results/combined_predicted_dag_sentences.tsv.xz"
        },
        "CtD":
        {
            "coco_score":"../../../compound_disease/compound_treats_disease/literature_models/coco_score/results/cd_edge_prediction_cocoscore.tsv",
            "our_model": "../../../compound_disease/compound_treats_disease/edge_prediction_experiment/results/combined_predicted_ctd_sentences.tsv.xz"
        },
        "CbG":
        {
            "coco_score":"../../../compound_gene/compound_binds_gene/literature_models/coco_score/results/cg_edge_prediction_cocoscore.tsv",
            "our_model": "../../../compound_gene/compound_binds_gene/edge_prediction_experiment/results/combined_predicted_cbg_sentences.tsv.xz"

        },
        "GiG":
        {
            "coco_score":"../../../gene_gene/gene_interacts_gene/literature_models/coco_score/results/gg_edge_prediction_cocoscore.tsv",
            "our_model": "../../../gene_gene/gene_interacts_gene/edge_prediction_experiment/results/combined_predicted_gig_sentences.tsv.xz"

        }
    })


    # Use the file tree above and graph the appropiate files
    edge_data_tree = OrderedDict({
        key: {
            sub_key:pd.read_csv(
                file_tree[key][sub_key],
                sep="\t"
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
        title="Hetionet Edge Prediction",
        file_name="../model_comparison.png", data=edge_data_tree,
        color_map=color_map
    )
