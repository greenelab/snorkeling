
# coding: utf-8

# # Re-Organize the Candidates

# From the [previous notebook](1.data-loader.ipynb) we aim to stratify the candidates into the appropiate categories (training, development, test). This part is easy because the only intensive operation is to update rows in a database. 

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

#Imports
import csv
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


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


from snorkel.models import  candidate_subclass, Candidate


# # Make All Possible Disease-Gene Pairs

# In this section of the notebook we plan to take the cartesian product between disease ontology terms and entrez gene terms. This product will contain all possible pair mapping between diseases and genes.

# In[ ]:


#url = 'https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv'
#disease_ontology_df = pd.read_csv(url, sep="\t")
#disease_ontology_df = (
#    disease_ontology_df
#    .drop_duplicates(["doid_code", "doid_name"])
#    .rename(columns={'doid_code': 'doid_id'})
#)


# In[4]:


url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
compound_df = pd.read_csv(url, sep="\t")
compound_df.head(2)


# In[5]:


url = 'https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv'
gene_entrez_df = pd.read_table(url, dtype={'GeneID': str})
gene_entrez_df = (
    gene_entrez_df
    [["GeneID", "Symbol"]]
    .rename(columns={'GeneID': 'entrez_gene_id', 'Symbol': 'gene_symbol'})
)
gene_entrez_df.head(2)


# In[6]:


gene_entrez_df['dummy_key'] = 0
#disease_ontology_df['dummy_key'] = 0
compound_df['dummy_key'] = 0
pair_df = gene_entrez_df.merge(compound_df[["drugbank_id", "name", "dummy_key"]], on='dummy_key').drop('dummy_key', axis=1)
pair_df.head(2)


# In[7]:


pair_df.to_csv("data/compound-gene-pairs-binds-full-map.csv", index=False, float_format='%.5g')


# ## Label All Pairs Whether or Not They are in Hetnets

# Here is where determine which disease - gene pair are located in hetionet. Pairs that have a source as a reference are considered to be apart of hetionet. 

# In[ ]:


#url = "https://github.com/dhimmel/integrate/raw/93feba1765fbcd76fd79e22f25121f5399629148/compile/DaG-association.tsv"
url = "https://raw.githubusercontent.com/dhimmel/integrate/93feba1765fbcd76fd79e22f25121f5399629148/compile/CbG-binding.tsv"
dag_df = pd.read_table(url, dtype={'entrez_gene_id': str})
dag_df.head(2)


# In[ ]:


dg_map_df = pair_df.merge(dag_df[["drugbank_id", "entrez_gene_id", "sources"]], how='left')
dg_map_df['hetionet'] = dg_map_df.sources.notnull().astype(int)
dg_map_df.head(2)


# In[ ]:


dg_map_df.hetionet.value_counts()


# ## See If D-G Pair is in Pubmed

# In this section we determine if a disease-gene pair is in our database. The resulting dataframe will contain the total number of sentences that each pair may have in pubmed and a boolean to recognize sentences that are greater than or equal to 0.

# In[ ]:


query = '''
SELECT "Disease_cid" AS doid_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM disease_gene
GROUP BY "Disease_cid", "Gene_cid";
'''
sentence_count_df = pd.read_sql(query, database_str)
sentence_count_df.head(2)


# In[ ]:


dg_map_df = dg_map_df.merge(sentence_count_df, how='left')
dg_map_df.n_sentences = dg_map_df.n_sentences.fillna(0).astype(int)
dg_map_df['has_sentence'] = (dg_map_df.n_sentences > 0).astype(int)
dg_map_df.head(2)


# In[ ]:


dg_map_df.has_sentence.value_counts()


# # See if Compound-gene pair is in Pubmed

# In[8]:


query = '''
SELECT "Compound_cid" AS drugbank_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM compound_gene
GROUP BY "Compound_cid", "Gene_cid";
'''
sentence_count_df = (
    pd.read_sql(query, database_str)
    .astype(dtype={'entrez_gene_id': int})
)
sentence_count_df.head(2)


# In[9]:


url = "https://raw.githubusercontent.com/dhimmel/integrate/93feba1765fbcd76fd79e22f25121f5399629148/compile/CbG-binding.tsv"
dag_df = pd.read_table(url, dtype={'entrez_gene_id': int})
dag_df.head(2)


# In[13]:


for r in tqdm_notebook(pd.read_csv("data/compound-gene-pairs-binds-full-map.csv", chunksize=1e6, dtype={'entrez_gene_id': int})):
    merged_df = pd.merge(r, dag_df[["drugbank_id", "entrez_gene_id", "sources"]], how="left")
    merged_df['hetionet'] = merged_df.sources.notnull().astype(int)
    merged_df = merged_df.merge(sentence_count_df, how='left', copy=False)
    merged_df.n_sentences = merged_df.n_sentences.fillna(0).astype(int)
    merged_df['has_sentence'] = (merged_df.n_sentences > 0).astype(int)
    merged_df.to_csv("data/compound-gene-pairs-binds-mapping.csv", mode='a', index=False)


# In[14]:


os.system(
    "head -n 1 compound-gene-pairs-binds-mapping.csv > compound-gene-pairs-binds.csv;" +
    "cat compound-gene-pairs-binds-mapping.csv |  awk -F ',' '{if($8==1) print $0}' >> compound-gene-pairs-binds.csv"
)


# In[15]:


dg_map_df = pd.read_csv("data/compound-gene-pairs-binds.csv")
dg_map_df.head(2)


# ## Modify the Candidate split

# This code below changes the split column of the candidate table. This column is what separates each sentence candidate into the corresponding categories (training (0), dev (1), tes. 

# In[19]:


def partitioner(df):
    partition_rank = pd.np.linspace(0, 1, num=len(df), endpoint=False)
    pd.np.random.shuffle(partition_rank)
    df['partition_rank'] = partition_rank
    return df

pd.np.random.seed(100)
dg_map_df = dg_map_df.groupby(['hetionet', 'has_sentence']).apply(partitioner)
dg_map_df.head(2)


# In[20]:


def get_split(partition_rank, training=0.7, dev=0.2, test=0.1):
    """
    This function partitions the data into training (0), dev (1), and test (2) sets
    """
    if partition_rank < training:
        return 6
    partition_rank -= training
    if partition_rank < dev:
        return 7
    partition_rank -= dev
    assert partition_rank <= test
    return 8

dg_map_df['split'] = dg_map_df.partition_rank.map(get_split)
dg_map_df.split.value_counts()


# In[21]:


#dg_map_df.to_csv("data/disease-gene-pairs-association.csv.xz", index=False, float_format='%.5g', compression='xz')
dg_map_df.to_csv("data/compound-gene-pairs-binds.csv", index=False, float_format='%.5g')


# In[22]:


dg_map_df.sources.unique()


# ## Re-categorize The Candidates

# In[32]:


sql = '''
SELECT id, "Compound_cid" AS drugbank_id, "Gene_cid" AS entrez_gene_id 
FROM compound_gene
'''
candidate_df = (
    pd.read_sql(sql, database_str)
    .astype(dtype={'entrez_gene_id': int})
    .merge(dg_map_df, how='left')
    .assign(type='compound_gene')
    [["id", "type", "split"]]
    .dropna(axis=0)
)
candidate_df.head(2)


# In[35]:


candidate_df.split.value_counts()


# In[36]:


candidate_df.shape


# ### Update Candidate table in database with splits

# In[37]:


get_ipython().run_cell_magic('time', '', "session.bulk_update_mappings(\n    Candidate,\n    candidate_df.to_dict(orient='records')\n)")


# In[38]:


from pandas.testing import assert_frame_equal
sql = '''
SELECT * FROM candidate
WHERE type = 'compound_gene';
'''
db_df = pd.read_sql(sql, database_str).sort_values('id')
compare_df = db_df.merge(candidate_df, on=['id', 'type'])
(compare_df.split_x == compare_df.split_y).value_counts()


# In[39]:


db_df.split.value_counts()

