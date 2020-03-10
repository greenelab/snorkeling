#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import sys
from tqdm import tqdm_notebook

sys.path.append(os.path.abspath('../../modules'))

from utils.notebook_utils.dataframe_helper import mark_sentence


# In[2]:


#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()


# In[3]:


from snorkel.learning.pytorch.rnn.utils import candidate_to_tokens
from snorkel.models import Candidate, candidate_subclass


# In[4]:


def get_edge_predictions(df):
    agg_df=(
        df
        .groupby(["source_node", "target_node"])
        .agg({"disc_model_prediction":["max", "idxmax"]})
        .reset_index()
    )
    agg_df.columns = [
        "_".join(col) 
        if col[1] != '' else col[0] 
        for col in agg_df.columns.values
    ]
    
    return (
        df
        .iloc[agg_df.disc_model_prediction_idxmax]
        .sort_values("disc_model_prediction", ascending=False)
    )


# In[5]:


file_tree = {
    "DaG":
    {
        "gen_model": {
            "train":"../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/DaG/marginals/train/30_sampled_train.tsv.xz",
            "tune":"../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/DaG/marginals/tune/30_sampled_dev.tsv",
            "test":"../../../disease_gene/disease_associates_gene/label_sampling_experiment/results/DaG/marginals/test/30_sampled_test.tsv",
        },
        "disc_model": "../../../disease_gene/disease_associates_gene/edge_prediction_experiment/results/combined_predicted_dag_sentences.tsv.xz",
        "dataset_statistics":"../../../disease_gene/disease_associates_gene/dataset_statistics/results/all_dag_map.tsv.xz"
    },
    "CtD":
    {
        "gen_model": {
            "train":"../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/CtD/marginals/train/22_sampled_train.tsv.xz",
            "tune":"../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/CtD/marginals/tune/22_sampled_dev.tsv",
            "test":"../../../compound_disease/compound_treats_disease/label_sampling_experiment/results/CtD/marginals/test/22_sampled_test.tsv",
        },
        "disc_model": "../../../compound_disease/compound_treats_disease/edge_prediction_experiment/results/combined_predicted_ctd_sentences.tsv.xz",
        "dataset_statistics": "../../../compound_disease/compound_treats_disease/dataset_statistics/results/all_ctd_map.tsv.xz"
    },
    "CbG":
    {
        "gen_model": {
            "train":"../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/CbG/marginals/train/20_sampled_train.tsv.xz",
            "tune":"../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/CbG/marginals/tune/20_sampled_dev.tsv",
            "test":"../../../compound_gene/compound_binds_gene/label_sampling_experiment/results/CbG/marginals/test/20_sampled_test.tsv",
        },
        "disc_model": "../../../compound_gene/compound_binds_gene/edge_prediction_experiment/results/combined_predicted_cbg_sentences.tsv.xz",
        "dataset_statistics": "../../../compound_gene/compound_binds_gene/dataset_statistics/results/all_cbg_candidates.tsv.xz"
    },
    #"GiG":
    #{
    #    "gen_model": "../../../../gene_gene/gene_interacts_gene/label_sampling_experiment/results/GiG/marginals/test/28_sampled_test.tsv",
    #    "disc_model": "../../../gene_gene/gene_interacts_gene/edge_prediction_experiment/results/combined_predicted_gig_sentences.tsv.xz",
    #    "dataset_statistics":"../../../gene_gene/gene_interacts_gene/dataset_statistics/results/all_gig_candidates.tsv.xz"
    #}
}


# In[6]:


relation_data_dict = {}
for rel in file_tree:
    for model in file_tree[rel]:
        if model == "gen_model":
            gen_model = (
                pd.concat([
                    pd.read_csv(file_tree[rel][model][dataset], sep="\t").iloc[:,[0,-1]]
                    for dataset in file_tree[rel][model]
                ],
                    axis=0, 
                    ignore_index=True
                )
            )
            gen_model.columns = ["gen_model_prediction", "candidate_id"]
            
        elif model == "disc_model":
            disc_model = (
                pd.read_csv(file_tree[rel][model], sep="\t")
                .rename(index=str, columns={"model_prediction":"disc_model_prediction"})
            )
            
        else:
            data_stat = pd.read_csv(file_tree[rel][model], sep="\t")
            
    relation_data_dict[rel] = (
        disc_model
        .merge(gen_model, on="candidate_id")
        .merge(data_stat[["n_sentences", "candidate_id"]], on="candidate_id")
        .assign(edge_type=rel)
        .assign(hetionet=lambda x: x["hetionet"].apply(lambda y: "Existing" if y == 1 else "Novel"))
    )


# In[7]:


relation_data_dict["DaG"] = get_edge_predictions(
    relation_data_dict["DaG"]
    .drop(["doid_id", "entrez_gene_id"], axis=1)
    .rename(index=str, columns={
        "gene_symbol":"target_node",
        "doid_name":"source_node",
    })
)
relation_data_dict["DaG"].head(2)


# In[8]:


relation_data_dict["CtD"] = get_edge_predictions(
    relation_data_dict["CtD"]
    .drop(["doid_id", "drugbank_id"], axis=1)
    .rename(index=str, columns={
        "doid_name":"target_node",
        "drug_name":"source_node",
    })
)
relation_data_dict["CtD"].head(2)


# In[9]:


relation_data_dict["CbG"] = get_edge_predictions(
    relation_data_dict["CbG"]
    .drop(["drugbank_id", "entrez_gene_id"], axis=1)
    .rename(index=str, columns={
        "gene_symbol":"target_node",
        "drug_name":"source_node",
    })
)
relation_data_dict["CbG"].head(2)


# In[10]:


#relation_data_dict["GiG"] = (
#    relation_data_dict["GiG"]
#    .drop(["gene1_id", "gene2_id"], axis=1)
#    .rename(index=str, columns={
#        "gene2_name":"target_node",
#        "gene1_name":"source_node",
#    })
#)
#relation_data_dict["GiG"].head(2)


# In[11]:


def tag_sentence(x, cand_class):
    candidates=(
        session
        .query(cand_class)
        .filter(cand_class.id.in_(x.candidate_id.astype(int).tolist()))
        .all()
    )
    tagged_sen=[
         " ".join(
             mark_sentence(
                candidate_to_tokens(cand), 
                [
                        (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                        (cand[1].get_word_start(), cand[1].get_word_end(), 2)
                ]
            )
         )
        for cand in candidates
    ]

    return tagged_sen


# In[12]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])
CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
GeneGene = candidate_subclass('GeneGene', ["Gene1", "Gene2"])


# In[13]:


for rel in relation_data_dict:
    if rel == "DaG":
        relation_data_dict[rel] = (
            relation_data_dict[rel]
            .head(10)
            .sort_values("candidate_id")
            .assign(text=lambda x: tag_sentence(x, DiseaseGene))
            .sort_values("disc_model_prediction", ascending=False)
        )
    elif rel == "CtD":
        relation_data_dict[rel] = (
            relation_data_dict[rel]
            .head(10)
            .sort_values("candidate_id")
            .assign(text=lambda x: tag_sentence(x, CompoundDisease))
            .sort_values("disc_model_prediction", ascending=False)
        )
    elif rel == "CbG":
        relation_data_dict[rel] = (
            relation_data_dict[rel]
            .head(10)
            .sort_values("candidate_id")
            .assign(text=lambda x: tag_sentence(x, CompoundGene))
            .sort_values("disc_model_prediction", ascending=False)
        )
    else:
        relation_data_dict[rel] = (
            relation_data_dict[rel]
            .head(10)
            .sort_values("candidate_id")
            .assign(text=lambda x: tag_sentence(x, GeneGene))
            .sort_values("disc_model_prediction", ascending=False)
        )


# In[15]:


total_table_df = pd.concat([
    relation_data_dict[rel] 
    for rel in relation_data_dict],
    axis=0, 
    ignore_index=True,
    sort=False
)
total_table_df


# In[17]:


(
    total_table_df
    [[
        "edge_type",
        "source_node", "target_node", 
        "hetionet",
        "gen_model_prediction", "disc_model_prediction",
        "n_sentences", "text"
    ]]
    .to_csv("../generative_model_predictions.tsv", sep='\t', index=False, float_format='%.3f')
)

