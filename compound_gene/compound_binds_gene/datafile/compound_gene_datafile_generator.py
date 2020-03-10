#!/usr/bin/env python
# coding: utf-8

# # Generate Compound Binds Gene Candidates

# This notebook is designed to construct a table that contains compound and gene pairs with various statistics (number of sentences, if contained in hetionet, if the edge has sentences and which training category each pair belongs to).

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


compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
cbg_url = "https://raw.githubusercontent.com/dhimmel/integrate/93feba1765fbcd76fd79e22f25121f5399629148/compile/CbG-binding.tsv"


# ## Read in Gene and Compound Entities

# In[4]:


entrez_gene_df = pd.read_table(gene_url).rename(index=str, columns={"GeneID": "entrez_gene_id", "Symbol":"gene_symbol"})
entrez_gene_df.head(2)


# In[5]:


drugbank_df = pd.read_table(compound_url).rename(index=str, columns={'name':'drug_name'})
drugbank_df.head(2)


# ## Read in Compound Binds/Regulates Gene Tables

# In[6]:


compound_binds_gene_df = pd.read_table(cbg_url, dtype={'entrez_gene_id': int})
compound_binds_gene_df.head(2)


# ## Read in Sentences with Edge Pair

# In[7]:


query = '''
SELECT "Compound_cid" AS drugbank_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM compound_gene
GROUP BY "Compound_cid", "Gene_cid";
'''

compound_gene_sentence_df = (
    pd
    .read_sql(query, database_str)
    .astype({"entrez_gene_id":int})
    .merge(drugbank_df[["drugbank_id", "drug_name"]], on="drugbank_id")
    .merge(entrez_gene_df[["entrez_gene_id", "gene_symbol"]], on="entrez_gene_id")
)
compound_gene_sentence_df.head(2)


# ## Merge Edges Into a Unified Table

# In[8]:


compound_binds_gene_df = (
    compound_binds_gene_df
    .merge(compound_gene_sentence_df, on=["drugbank_id", "entrez_gene_id"], how="outer")
)
compound_binds_gene_df=(
    compound_binds_gene_df
    .assign(hetionet=compound_binds_gene_df.sources.notnull().astype(int))
    .assign(has_sentence=(compound_binds_gene_df.n_sentences > 0).astype(int))
)
compound_binds_gene_df.head(2)


# In[9]:


# Make sure all existing edges are found
# 11571 is determined from neo4j to be all DaG Edges
assert compound_binds_gene_df.hetionet.value_counts()[1] == 24687


# In[10]:


compound_binds_gene_df.query("hetionet==1&has_sentence==1").shape


# Make note that 18741 edges in Hetionet do not have sentences

# ## Sort Edges into categories

# In[11]:


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


# In[12]:


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
        return 6
    partition_rank -= training
    if partition_rank < dev:
        return 7
    partition_rank -= dev
    assert partition_rank <= test
    return 8


# In[13]:


pd.np.random.seed(100)
cbg_map_df = compound_binds_gene_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
cbg_map_df.head(2)


# In[14]:


cbg_map_df['split'] = cbg_map_df.partition_rank.map(get_split)
cbg_map_df.split.value_counts()


# In[15]:


cbg_map_df.sources.unique()


# In[16]:


cbg_map_df = cbg_map_df[[
    "drugbank_id", "drug_name",
    "entrez_gene_id", "gene_symbol",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
cbg_map_df.head(2)


# In[17]:


cbg_map_df.to_csv("output/compound_binds_gene.tsv.xz", sep="\t", compression="xz", index=False)

