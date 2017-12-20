
# coding: utf-8

# # Re-Organize the Candidates

# From the [previous notebook](1.data-loader.ipynb) we aim to stratify the candidates into the appropiate categories (training, development, test). Since the hard work (data insertion) was already done, this part is easy as it breaks down into relabeling the split column inside the Candidate table. The split column will be used throughout the rest of this pipeline.

# In[2]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

#Imports
import csv
import os
import random

import numpy as np
import pandas as pd
import tqdm


# In[3]:


#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()


# In[4]:


from snorkel.models import  candidate_subclass


# In[5]:


#This specifies the type of candidates to extract
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# # Make Stratified File

# In[ ]:


disease_ontology_df = pd.read_csv('https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv', sep="\t")
disease_ontology_df = disease_ontology_df.drop_duplicates(["doid_code", "doid_name"])


# In[ ]:


gene_entrez_df = pd.read_csv('https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv', sep="\t")
gene_entrez_df = gene_entrez_df[["GeneID", "Symbol"]]


# ## Map Each Disease to Each Gene

# In[ ]:


gene_entrez_df['dummy_key'] =0
disease_ontology_df['dummy_key'] = 0
dg_map_df = gene_entrez_df.merge(disease_ontology_df[["doid_code", "doid_name", "dummy_key"]], on='dummy_key')


# ## Label All Pairs Whether or Not They are in Hetnets

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'hetnet_kb_df = pd.read_csv("hetnet_dg_kb.csv")\nhetnet_set = set(map(lambda x: tuple(x), hetnet_kb_df.values))\nhetnet_labels = np.ones(dg_map_df.shape[0]) * -1\n\nfor index, row in tqdm.tqdm(dg_map_df.iterrows()):\n    if (row["doid_code"], row["GeneID"]) in hetnet_set:\n        hetnet_labels[index] = 1 \n    \ndg_map_df["hetnet"] = hetnet_labels')


# ## See if D-G Pair is in Pubmed

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'pubmed_dg_pairs = set({})\ncands = []\nchunk_size = 1e5\noffset = 0\n\nwhile True:\n    cands = session.query(DiseaseGene).limit(chunk_size).offset(offset).all()\n    \n    if not cands:\n        break\n        \n    for candidate in tqdm.tqdm(cands):\n        pubmed_dg_pairs.add((candidate.Disease_cid, candidate.Gene_cid))\n    \n    offset = offset + chunk_size')


# In[ ]:


pubmed_labels = np.ones(dg_map_df.shape[0]) * -1

for index, row in tqdm.tqdm(dg_map_df.iterrows()):
    if (row["doid_code"], str(row["GeneID"])) in pubmed_dg_pairs:
        pubmed_labels[index] = 1

dg_map_df["pubmed"] = pubmed_labels


# In[ ]:


dg_map_df = dg_map_df.rename(index=str, columns={"GeneID": "gene_id", "doid_code": "disease_id", "doid_name": "disease_name", "Symbol":"gene_name"})
dg_map_df["hetnet"] = dg_map_df["hetnet"].astype(int)
dg_map_df["pubmed"] = dg_map_df["pubmed"].astype(int)
dg_map_df.to_csv("dg_map.csv", index=False)


# ## Modify the Candidate split

# This code below changes the split column of the candidate table as mentioned above. Using sqlalchemy and the chunking strategy, every candidate that has the particular disease entity id (DOID:3393) will be given the category of 2. 2 Representes the testing set which will be used in the rest of the notebooks.

# In[6]:


dg_map_df = pd.read_csv("dg_map.csv")


# In[7]:


print dg_map_df[(dg_map_df["hetnet"] == 1)].shape
print dg_map_df[(dg_map_df["pubmed"]== 1)].shape
print
print dg_map_df[(dg_map_df["hetnet"] == 1)&(dg_map_df["pubmed"]== 1)].shape
print dg_map_df[(dg_map_df["hetnet"] == 1)&(dg_map_df["pubmed"]== -1)].shape
print dg_map_df[(dg_map_df["hetnet"] == -1)&(dg_map_df["pubmed"]== 1)].shape
print dg_map_df[(dg_map_df["hetnet"] == -1)&(dg_map_df["pubmed"]== -1)].shape


# In[20]:


test_size = 0.1
dev_size = 0.2
training_size = 0.7
random_seed = 100

sizes = []
sizes.append(dg_map_df[(dg_map_df["hetnet"] == 1)&(dg_map_df["pubmed"]== 1)].shape[0])
sizes.append(dg_map_df[(dg_map_df["hetnet"] == 1)&(dg_map_df["pubmed"]== -1)].shape[0])
sizes.append(dg_map_df[(dg_map_df["hetnet"] == -1)&(dg_map_df["pubmed"]== 1)].shape[0])
sizes.append(dg_map_df[(dg_map_df["hetnet"] == -1)&(dg_map_df["pubmed"]== -1)].shape[0])

dummy_dg_map = dg_map_df

for data_size, file_name in zip([test_size, dev_size], ["stratified_data/test_set.csv", "stratified_data/dev_set.csv"]):
    adjusted_size = np.round(np.array(sizes) * data_size).astype(int)

    hetnet_pubmed = dummy_dg_map[(dummy_dg_map["hetnet"] == 1)&(dummy_dg_map["pubmed"]== 1)].sample(adjusted_size[0], random_state=random_seed)
    hetnet_no_pubmed = dummy_dg_map[(dummy_dg_map["hetnet"] == 1)&(dummy_dg_map["pubmed"]== -1)].sample(adjusted_size[1], random_state=random_seed)
    no_hetnet_pubmed = dummy_dg_map[(dummy_dg_map["hetnet"] == -1)&(dummy_dg_map["pubmed"]== 1)].sample(adjusted_size[2], random_state=random_seed)
    no_hetnet_no_pubmed = dummy_dg_map[(dummy_dg_map["hetnet"] == -1)&(dummy_dg_map["pubmed"]== -1)].sample(10000, random_state=random_seed)
    
    final_dataset = hetnet_pubmed.append(hetnet_no_pubmed).append(no_hetnet_pubmed).append(no_hetnet_no_pubmed)
    final_dataset.to_csv(file_name, index=False)
    dummy_dg_map = dummy_dg_map.drop(final_dataset.index)

final_dataset = dummy_dg_map[(dummy_dg_map["hetnet"] == 1)&(dummy_dg_map["pubmed"]== 1)]
final_dataset = final_dataset.append(dummy_dg_map[(dummy_dg_map["hetnet"] == -1)&(dummy_dg_map["pubmed"]== 1)])
final_dataset = final_dataset.append(dummy_dg_map[(dummy_dg_map["hetnet"] == 1)&(dummy_dg_map["pubmed"]== -1)])
final_dataset = final_dataset.append(dummy_dg_map[(dummy_dg_map["hetnet"] == -1)&(dummy_dg_map["pubmed"]== -1)].sample(10000, random_state=random_seed))
final_dataset.to_csv("stratified_data/training_set.csv", index=False)


# ## Re-categorize The Candidates

# In[8]:


test_df = pd.read_csv("stratified_data/test_set.csv")
test_set = set(map(tuple, test_df[(test_df["pubmed"] == 1)][["disease_id","gene_id"]].values))

dev_df = pd.read_csv("stratified_data/dev_set.csv")
dev_set = set(map(tuple, dev_df[(dev_df["pubmed"] == 1)][["disease_id","gene_id"]].values))


# In[9]:


get_ipython().run_cell_magic(u'time', u'', u'cands = []\nchunk_size = 1e5\noffset = 0\n\nwhile True:\n    cands = session.query(DiseaseGene).limit(chunk_size).offset(offset).all()\n    \n    if not cands:\n        break\n        \n    for candidate in tqdm.tqdm(cands):\n        if (candidate.Disease_cid, int(candidate.Gene_cid)) in test_set:\n            candidate.split = 2\n        elif (candidate.Disease_cid, int(candidate.Gene_cid)) in dev_set:\n            candidate.split = 1\n        else:\n            candidate.split = 0\n        \n        session.add(candidate)\n    \n    offset = offset + chunk_size\n# persist the changes into the database\nsession.commit()')

