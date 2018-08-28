
# coding: utf-8

# # Re-Organize the Candidates

# From the [previous notebook](1.data-loader.ipynb) we aim to stratify the candidates into the appropiate categories (training, development, test). This part is easy because the only intensive operation is to update rows in a database. 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

#Imports
import csv
import os
import random

import numpy as np
import pandas as pd


# In[ ]:


#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()


# In[ ]:


from snorkel.models import  candidate_subclass, Candidate


# ## Modify the Candidate split

# In[ ]:


#from utils.datafiles.disease_gene_datafiles import dag_map_df as map_df
from utils.datafiles.compound_gene_datafiles import cbg_map_df as map_df


# This code below changes the split column of the candidate table. This column is what separates each sentence candidate into the corresponding categories (training, dev, test). 

# In[ ]:


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


# In[ ]:


pd.np.random.seed(100)
map_df = dg_map_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
map_df.head(2)


# In[ ]:


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


# In[ ]:


map_df['split'] = dg_map_df.partition_rank.map(get_split)
map_df.split.value_counts()


# In[ ]:


map_df.sources.unique()


# In[ ]:


map_df.to_csv("data/compound_gene/compound_binds_gene/compound_gene_pairs_binds.csv", index=False, float_format='%.5g')


# ## Re-categorize The Candidates

# In[ ]:


sql = '''
SELECT id, "Compound_cid" AS drugbank_id, "Gene_cid" AS entrez_gene_id 
FROM compound_gene
'''
candidate_df = (
    pd.read_sql(sql, database_str)
    .astype(dtype={'entrez_gene_id': int})
    .merge(map_df, how='left')
    .assign(type='compound_gene')
    [["id", "type", "split"]]
    .dropna(axis=0)
)
candidate_df.head(2)


# In[ ]:


candidate_df.split.value_counts()


# In[ ]:


candidate_df.shape


# ### Update Candidate table in database with splits

# In[ ]:


get_ipython().run_cell_magic('time', '', "session.bulk_update_mappings(\n    Candidate,\n    candidate_df.to_dict(orient='records')\n)")


# In[ ]:


from pandas.testing import assert_frame_equal
sql = '''
SELECT * FROM candidate
WHERE type = 'compound_gene';
'''
db_df = pd.read_sql(sql, database_str).sort_values('id')
compare_df = db_df.merge(candidate_df, on=['id', 'type'])
(compare_df.split_x == compare_df.split_y).value_counts()


# In[ ]:


db_df.split.value_counts()

