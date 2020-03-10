#!/usr/bin/env python
# coding: utf-8

# # Generate Gene Interacts Gene Candidates

# This notebook is designed to construct a table that contains gene pairs with various statistics (number of sentences, if contained in hetionet, if the edge has sentences and which training category each pair belongs to).

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


gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
ppi_url = "https://raw.githubusercontent.com/dhimmel/ppi/f6a7edbc8de6ba2d7fe1ef3fee4d89e5b8d0b900/data/ppi-hetio-ind.tsv"


# ## Read in Gene Entities

# In[4]:


entrez_gene_df = pd.read_table(gene_url).rename(index=str, columns={"GeneID": "entrez_gene_id"})
entrez_gene_df.head(2)


# ## Read in Gene Interacts Gene Table

# In[5]:


gene_gene_interaction_df = pd.read_table(ppi_url)
gene_gene_interaction_df.head(2)


# ## Read in Sentences with Edge Pair

# In[8]:


query = '''
SELECT "Gene1_cid" AS gene1_id, "Gene2_cid" AS gene2_id, count(*) AS n_sentences
FROM gene_gene
GROUP BY "Gene1_cid", "Gene2_cid";
'''
gene_gene_sentence_df = pd.read_sql(query, database_str).astype({"gene1_id":int, "gene2_id":int})
gene_gene_sentence_df.head(2)


# ## Merge Edges Into a Unified Table

# In[9]:


gene_gene_interaction_df = (
    gene_gene_interaction_df
    .rename(index=str, columns={"gene_0":"gene1_id", "gene_1":"gene2_id"})
    .merge(gene_gene_sentence_df, on=["gene1_id", "gene2_id"], how="right")
)
gene_gene_interaction_df=(
    gene_gene_interaction_df
    .assign(hetionet=gene_gene_interaction_df.sources.notnull().astype(int))
    .assign(has_sentence=(gene_gene_interaction_df.n_sentences > 0).astype(int))
)
gene_gene_interaction_df.head(2)


# In[8]:


gene_gene_interaction_df=(
    gene_gene_interaction_df
    .merge(
        entrez_gene_df[["entrez_gene_id", "Symbol"]]
        .rename(index=str, columns={"entrez_gene_id": "gene1_id", "Symbol":"gene1_name"}), 
        on="gene1_id"
    )
    .merge(
        entrez_gene_df[["entrez_gene_id", "Symbol"]]
        .rename(index=str, columns={"entrez_gene_id": "gene2_id", "Symbol":"gene2_name"}), 
        on="gene2_id"
    )
)
gene_gene_interaction_df.head(2)


# ## Sort Edges into categories

# In[9]:


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


# In[10]:


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
        return 3
    partition_rank -= training
    if partition_rank < dev:
        return 4
    partition_rank -= dev
    assert partition_rank <= test
    return 5


# In[11]:


pd.np.random.seed(100)
map_df = gene_gene_interaction_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
map_df.head(2)


# In[12]:


map_df['split'] = map_df.partition_rank.map(get_split)
map_df.split.value_counts()


# In[13]:


map_df.sources.unique()


# In[14]:


map_df = map_df[[
    "gene1_id", "gene1_name",
    "gene2_id", "gene2_name",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
map_df.head(2)


# In[15]:


map_df.to_csv("output/gene_interacts_gene.tsv.xz", sep="\t", compression="xz", index=False)

