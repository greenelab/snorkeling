#!/usr/bin/env python
# coding: utf-8

# # Generate Disease Associates Gene Candidates

# This notebook is designed to construct a table that contains disease and gene pairs with various statistics (number of sentences, if contained in hetionet, if the edge has sentences and which training category each pair belongs to).

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter
from itertools import product
import os
import pickle
import sys

import pandas as pd


# In[2]:


#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)


# In[3]:


disease_url = "https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
dag_url = "https://github.com/dhimmel/integrate/raw/93feba1765fbcd76fd79e22f25121f5399629148/compile/DaG-association.tsv"
drg_url = "https://raw.githubusercontent.com/dhimmel/stargeo/08b126cc1f93660d17893c4a3358d3776e35fd84/data/diffex.tsv"


# ## Read in Diesease and Gene Entities

# In[4]:


disease_ontology_df = (
    pd.read_csv(disease_url, sep="\t")
    .drop_duplicates(["doid_code", "doid_name"])
    .rename(columns={'doid_code': 'doid_id'})
)
disease_ontology_df.head(2)


# In[5]:


entrez_gene_df = (
    pd.read_csv(gene_url, sep="\t")
    .rename(index=str, columns={"GeneID": "entrez_gene_id", "Symbol":"gene_symbol"})
)
entrez_gene_df.head(2)


# ## Read in Disease Associates/Regulates Gene Tables

# In[6]:


disease_associates_gene_df = (
    pd.read_csv(dag_url, sep="\t", dtype={'entrez_gene_id': int})
)
disease_associates_gene_df.head(2)


# ## Read in Sentences with Edge Pair

# In[7]:


query = '''
SELECT "Disease_cid" AS doid_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM disease_gene
GROUP BY "Disease_cid", "Gene_cid";
'''
disease_gene_sentence_df = pd.read_sql(query, database_str).astype({"entrez_gene_id":int})
disease_gene_sentence_df.head(2)


# ## Merge Edges Into a Unified Table

# In[8]:


disease_gene_map_df = (
    entrez_gene_df[["entrez_gene_id", "gene_symbol"]]
    .assign(key=1)
    .merge(disease_ontology_df[["doid_id", "doid_name"]].assign(key=1))
    .drop("key", axis=1)
)
disease_gene_map_df.head(2)


# In[9]:


disease_gene_associations_df = (
    disease_gene_map_df
    .merge(
        disease_associates_gene_df
        [["doid_id", "entrez_gene_id", "sources"]],
        on=["doid_id", "entrez_gene_id"],
        how="left"
    )
    .merge(disease_gene_sentence_df, on=["doid_id", "entrez_gene_id"], how="left")
    .fillna({"n_sentences": 0})
    .astype({"n_sentences": int})
)
disease_gene_associations_df = (
    disease_gene_associations_df
    .assign(hetionet=disease_gene_associations_df.sources.notnull().astype(int))
    .assign(has_sentence=(disease_gene_associations_df.n_sentences > 0).astype(int))
)
disease_gene_associations_df.head(2)


# In[10]:


# Make sure all existing edges are found
# 12623 is determined from neo4j to be all DaG Edges
assert disease_gene_associations_df.hetionet.value_counts()[1] == 12623


# In[11]:


disease_gene_associations_df.query("hetionet==1&has_sentence==1").shape


# Make Note that 3044 number of edges do not contain sentences.

# ## Sort Edges into categories

# In[12]:


def partitioner(df):
    """
    This function creates a parition rank for the current dataset.
    This algorithm assigns a rank [0-1) for each datapoint inside each group (outlined below):
        1,1 -in hetionet and has sentences
        1,0 - in hetionet and doesn't have sentences
        0,1 - not in hetionet and does have sentences
        0,0, - not in hetionet and doesn't have sentences
        
    This ranking will be used in the get split function to assign each datapoint 
    into its corresponding category (train, dev, test)
    """
    partition_rank = pd.np.linspace(0, 1, num=len(df), endpoint=False)
    pd.np.random.shuffle(partition_rank)
    df['partition_rank'] = partition_rank
    return df


# In[13]:


def get_split(partition_rank, training=0.7, dev=0.2, test=0.1):
    """
    This function partitions the data into training, dev, and test sets
    The partitioning algorithm is as follows:
        1. anything less than 0.7 goes into training and receives an appropiate label
        2. If not less than 0.7 subtract 0.7 and see if the rank is less than 0.2 if not assign to dev
        3. Lastly if the rank is greater than 0.9 (0.7+0.2) assign it to test set.
        
    return label that corresponds to appropiate dataset cateogories
    """
    if partition_rank < training:
        return 1
    partition_rank -= training
    if partition_rank < dev:
        return 2
    partition_rank -= dev
    assert partition_rank <= test
    return 3


# In[14]:


pd.np.random.seed(100)
dag_map_df = disease_gene_associations_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
dag_map_df.head(2)


# In[15]:


dag_map_df['split'] = dag_map_df.partition_rank.map(get_split)
dag_map_df.split.value_counts()


# In[16]:


dag_map_df.sources.unique()


# In[17]:


dag_map_df.to_csv("output/disease_associates_gene.tsv.xz", sep="\t", compression="xz", index=False)

