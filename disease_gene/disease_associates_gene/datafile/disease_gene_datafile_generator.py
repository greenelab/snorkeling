
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


# In[5]:


entrez_gene_df = (
    pd.read_table(gene_url)
    .rename(index=str, columns={"GeneID": "entrez_gene_id", "Symbol":"gene_symbol"})
)
entrez_gene_df.head(2)


# ## Read in Disease Associates/Regulates Gene Tables

# In[6]:


disease_associates_gene_df = (
    pd.read_table(dag_url, dtype={'entrez_gene_id': int})
)
disease_associates_gene_df.head(2)


# In[7]:


disease_regulates_gene_df = (
    pd.read_table(drg_url, dtype={'entrez_gene_id': int})
    .assign(sources='strego')
    .rename(index=str, columns={'slim_id':'doid_id', 'slim_name':'doid_name'})
    .drop(["log2_fold_change", "p_adjusted"], axis=1)
)
disease_regulates_gene_df.head(2)


# ## Read in Sentences with Edge Pair

# In[8]:


query = '''
SELECT "Disease_cid" AS doid_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM disease_gene
GROUP BY "Disease_cid", "Gene_cid";
'''
disease_gene_sentence_df = pd.read_sql(query, database_str).astype({"entrez_gene_id":int})
disease_gene_sentence_df.head(2)


# ## Merge Edges Into a Unified Table

# In[9]:


disease_gene_associations_df = (
    disease_associates_gene_df
    .merge(disease_gene_sentence_df, on=["doid_id", "entrez_gene_id"], how="right")
)
disease_gene_associations_df = (
    disease_gene_associations_df
    .assign(hetionet=disease_gene_associations_df.sources.notnull().astype(int))
    .assign(has_sentence=(disease_gene_sentence_df.n_sentences > 0).astype(int))
)
disease_gene_associations_df.head(2)


# In[10]:


disease_gene_regulation_df = (
    disease_regulates_gene_df
    .merge(disease_gene_sentence_df, on=["doid_id", "entrez_gene_id"], how="right")
)
disease_gene_regulation_df = (
    disease_gene_regulation_df
    .assign(hetionet=disease_gene_regulation_df.sources.notnull().astype(int))
    .assign(has_sentence=(disease_gene_regulation_df.n_sentences > 0).astype(int))
)
disease_gene_regulation_df.head(2)


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
        return 1
    partition_rank -= training
    if partition_rank < dev:
        return 2
    partition_rank -= dev
    assert partition_rank <= test
    return 3


# In[13]:


pd.np.random.seed(100)
dag_map_df = disease_gene_associations_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
dag_map_df.head(2)


# In[14]:


dag_map_df['split'] = dag_map_df.partition_rank.map(get_split)
dag_map_df.split.value_counts()


# In[15]:


dag_map_df.sources.unique()


# In[16]:


dag_map_df.to_csv("results/disease_associates_gene.tsv.xz", sep="\t", compression="xz", index=False)


# In[17]:


disease_downregulates_gene_df = (
    disease_gene_regulation_df
    .query("direction=='down'|direction.isnull()")
    .drop('direction', axis=1)
)


# In[18]:


pd.np.random.seed(100)
ddg_map_df = disease_downregulates_gene_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
ddg_map_df.head(2)


# In[19]:


ddg_map_df['split'] = ddg_map_df.partition_rank.map(get_split)
ddg_map_df.split.value_counts()


# In[20]:


ddg_map_df.sources.unique()


# In[21]:


ddg_map_df.to_csv("results/disease_downregulates_gene.tsv.xz", sep="\t", compression="xz", index=False)


# In[22]:


disease_upregulates_gene_df = (
    disease_gene_regulation_df
    .query("direction=='up'|direction.isnull()")
    .drop('direction', axis=1)
)


# In[23]:


pd.np.random.seed(100)
dug_map_df = disease_upregulates_gene_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
dug_map_df.head(2)


# In[24]:


dug_map_df['split'] = dug_map_df.partition_rank.map(get_split)
dug_map_df.split.value_counts()


# In[25]:


dug_map_df.sources.unique()


# In[26]:


dug_map_df.to_csv("results/disease_upregulates_gene.tsv.xz", sep="\t", compression="xz", index=False)

