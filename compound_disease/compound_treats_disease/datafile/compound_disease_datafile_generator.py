
# coding: utf-8

# # Generate Compound Treats Disease Candidates

# This notebook is designed to construct a table that contains compound and disease pairs with various statistics (number of sentences, if contained in hetionet, if the edge has sentences and which training category each pair belongs to).

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


disease_url = 'https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv'
compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
ctpd_url = "https://raw.githubusercontent.com/dhimmel/indications/11d535ba0884ee56c3cd5756fdfb4985f313bd80/catalog/indications.tsv"


# In[4]:


base_dir = os.path.join(os.path.dirname(os.getcwd()), 'compound_disease')


# ## Read in Diesease and Compound Entities

# In[5]:


disease_ontology_df = (
    pd.read_csv(disease_url, sep="\t")
    .drop_duplicates(["doid_code", "doid_name"])
    .rename(columns={'doid_code': 'doid_id'})
)
disease_ontology_df.head(2)


# In[6]:


drugbank_df = (
    pd.read_table(compound_url)
    .rename(index=str, columns={'name':'drug_name'})
)
drugbank_df.head(2)


# ## Read in Compound Treats/Palliates Disease Tables

# In[7]:


compound_treats_palliates_disease_df = (
    pd.read_table(ctpd_url)
    .assign(sources='pharmacotherapydb')
    .drop(["n_curators", "n_resources"], axis=1)
    .rename(index=str, columns={"drug": "drug_name", "disease":"disease_name"})
)
compound_treats_palliates_disease_df.head(2)


# ## Read in Sentences with Edge Pair

# In[8]:


query = '''
SELECT "Compound_cid" as drugbank_id, "Disease_cid" as doid_id, count(*) AS n_sentences
FROM compound_disease
GROUP BY "Compound_cid", "Disease_cid";
'''
compound_disease_sentence_df = pd.read_sql(query, database_str)
compound_disease_sentence_df.head(2)


# ## Merge Edges Into a Unified Table

# In[9]:


compound_treats_disease_df = (
    compound_treats_palliates_disease_df
    .query("category=='DM'")
    .merge(compound_disease_sentence_df, on=["drugbank_id", "doid_id"], how="right")
)
compound_treats_disease_df=(
    compound_treats_disease_df
    .assign(hetionet=compound_treats_disease_df.sources.notnull().astype(int))
    .assign(has_sentence=(compound_treats_disease_df.n_sentences > 0).astype(int))
)
compound_treats_disease_df.head(2)


# In[10]:


compound_palliates_disease_df = (
    compound_treats_palliates_disease_df
    .query("category=='SYM'")
    .merge(compound_disease_sentence_df, on=["drugbank_id", "doid_id"], how="right")
)
compound_palliates_disease_df=(
    compound_palliates_disease_df
    .assign(hetionet=compound_treats_disease_df.sources.notnull().astype(int))
    .assign(has_sentence=(compound_treats_disease_df.n_sentences > 0).astype(int))
)
compound_palliates_disease_df.head(2)


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
        return 3
    partition_rank -= training
    if partition_rank < dev:
        return 4
    partition_rank -= dev
    assert partition_rank <= test
    return 5


# In[13]:


pd.np.random.seed(100)
ctd_map_df = compound_treats_disease_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
ctd_map_df.head(2)


# In[14]:


ctd_map_df['split'] = ctd_map_df.partition_rank.map(get_split)
ctd_map_df.split.value_counts()


# In[15]:


ctd_map_df.sources.unique()


# In[16]:


ctd_map_df = ctd_map_df[[
    "drugbank_id", "drug_name",
    "doid_id", "disease_name",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
ctd_map_df.head(2)


# In[17]:


ctd_map_df.to_csv("results/compound_treats_disease.tsv.xz", sep="\t", compression="xz", index=False)


# In[18]:


pd.np.random.seed(100)
cpd_map_df = compound_palliates_disease_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
cpd_map_df.head(2)


# In[19]:


cpd_map_df['split'] = cpd_map_df.partition_rank.map(get_split)
cpd_map_df.split.value_counts()


# In[20]:


cpd_map_df.sources.unique()


# In[21]:


cpd_map_df = cpd_map_df[[
    "drugbank_id", "drug_name",
    "doid_id", "disease_name",
    "sources", "n_sentences",
    "hetionet", "has_sentence",
    "split", "partition_rank"
]]
cpd_map_df.head(2)


# In[22]:


cpd_map_df.to_csv("results/compound_palliates_disease.tsv.xz", sep="\t", compression="xz", index=False)

