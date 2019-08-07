
# coding: utf-8

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
crg_url = "https://raw.githubusercontent.com/dhimmel/lincs/bbc6812b7d19e98637b44373cdfc52f61bce6327/data/consensi/signif/dysreg-drugbank.tsv"


# In[4]:


entrez_gene_df = pd.read_table(gene_url).rename(index=str, columns={"GeneID": "entrez_gene_id", "Symbol":"gene_symbol"})
entrez_gene_df.head(2)


# In[5]:


drugbank_df = pd.read_table(compound_url).rename(index=str, columns={'name':'drug_name'})
drugbank_df.head(2)


# In[6]:


compound_binds_gene_df = pd.read_table(cbg_url, dtype={'entrez_gene_id': int})
compound_binds_gene_df.head(2)


# In[7]:


compound_regulates_gene_df = (
    pd.read_table(crg_url, dtype={'entrez_gene_id': int})
    .assign(sources='lincs')
    .drop(['z_score', 'status', 'nlog10_bonferroni_pval'], axis=1)
    .rename(index=str, columns={"perturbagen":'drugbank_id'})
)
compound_regulates_gene_df.head(2)


# In[8]:


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


# In[9]:


compound_binds_gene_df = (
    compound_binds_gene_df
    .merge(compound_gene_sentence_df, on=["drugbank_id", "entrez_gene_id"], how="right")
)
compound_binds_gene_df=(
    compound_binds_gene_df
    .assign(hetionet=compound_binds_gene_df.sources.notnull().astype(int))
    .assign(has_sentence=(compound_binds_gene_df.n_sentences > 0).astype(int))
)
compound_binds_gene_df.head(2)


# In[10]:


compound_downregulates_gene_df = (
    compound_regulates_gene_df
    .query("direction=='down'")
    .merge(compound_gene_sentence_df, on=["drugbank_id", "entrez_gene_id"], how="right")
)
compound_downregulates_gene_df=(
    compound_downregulates_gene_df
    .assign(hetionet=compound_downregulates_gene_df.sources.notnull().astype(int))
    .assign(has_sentence=(compound_downregulates_gene_df.n_sentences > 0).astype(int))
)
compound_downregulates_gene_df.head(2)


# In[11]:


compound_upregulates_gene_df = (
    compound_regulates_gene_df
    .query("direction=='up'")
    .merge(compound_gene_sentence_df, on=["drugbank_id", "entrez_gene_id"], how="right")
)
compound_upregulates_gene_df=(
    compound_upregulates_gene_df
    .assign(hetionet=compound_upregulates_gene_df.sources.notnull().astype(int))
    .assign(has_sentence=(compound_upregulates_gene_df.n_sentences > 0).astype(int))
)
compound_upregulates_gene_df.head(2)


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
        return 6
    partition_rank -= training
    if partition_rank < dev:
        return 7
    partition_rank -= dev
    assert partition_rank <= test
    return 8


# In[14]:


pd.np.random.seed(100)
cbg_map_df = compound_binds_gene_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
cbg_map_df.head(2)


# In[15]:


cbg_map_df['split'] = cbg_map_df.partition_rank.map(get_split)
cbg_map_df.split.value_counts()


# In[16]:


cbg_map_df.sources.unique()


# In[17]:


cbg_map_df = cbg_map_df[[
    "drugbank_id", "drug_name",
    "entrez_gene_id", "gene_symbol",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
cbg_map_df.head(2)


# In[18]:


cbg_map_df.to_csv("results/compound_binds_gene.tsv.xz", sep="\t", compression="xz", index=False)


# In[19]:


pd.np.random.seed(100)
cdg_map_df = compound_downregulates_gene_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
cdg_map_df.head(2)


# In[20]:


cdg_map_df['split'] = cdg_map_df.partition_rank.map(get_split)
cdg_map_df.split.value_counts()


# In[21]:


cdg_map_df.sources.unique()


# In[22]:


cdg_map_df = cdg_map_df[[
    "drugbank_id", "drug_name",
    "entrez_gene_id", "gene_symbol",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
cdg_map_df.head(2)


# In[23]:


cdg_map_df.to_csv("results/compound_downregulates_gene.tsv.xz", sep="\t", compression="xz", index=False)


# In[24]:


pd.np.random.seed(100)
cug_map_df = compound_upregulates_gene_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
cug_map_df.head(2)


# In[25]:


cug_map_df['split'] = cbg_map_df.partition_rank.map(get_split)
cug_map_df.split.value_counts()


# In[26]:


cug_map_df.sources.unique()


# In[27]:


cug_map_df = cug_map_df[[
    "drugbank_id", "drug_name",
    "entrez_gene_id", "gene_symbol",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
cug_map_df.head(2)


# In[28]:


cbg_map_df.to_csv("results/compound_upregulates_gene.tsv.xz", sep="\t", compression="xz", index=False)

