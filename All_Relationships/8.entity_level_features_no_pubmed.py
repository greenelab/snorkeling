
# coding: utf-8

# # Generate Features For Entities Not in Pubmed

# This notebook is designed to calculate features for entities that are not mentioned in the Pubmed database. The features boil down to a pvalue of 1, a prior probability, and how many times a disease/gene is mentioned individually.

# In[ ]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

from collections import defaultdict
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy.stats import fisher_exact
import scipy
from sqlalchemy import and_
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns


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


from snorkel.models import Candidate, candidate_subclass
from snorkel.learning.disc_models.rnn import reRNN


# In[ ]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# # Count the Number of Sentences for Each Candidate

# For this block of code we are cycling through each disease-gene candidate in the database and counting the number of unique sentences and unique abstracts containing the specific candidate. NOTE: This section will quite a few hours to cycle through the entire database.

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'pair_to_pmids = defaultdict(set)\npair_to_sentences = defaultdict(set)\noffset = 0\nchunk_size = 1e5\n\nwhile True:\n    cands = session.query(DiseaseGene).limit(chunk_size).offset(offset).all()\n    \n    if not cands:\n        break\n        \n    for candidate in cands:\n        pair = candidate.Disease_cid, candidate.Gene_cid\n        pair_to_sentences[pair].add(candidate[0].get_parent().id)\n        pair_to_pmids[pair].add(candidate[0].get_parent().document_id)\n\n    offset+= chunk_size')


# In[ ]:


candidate_df = pd.DataFrame(
    map(lambda x: [x[0], x[1], len(pair_to_sentences[x]), len(pair_to_pmids[x])], pair_to_sentences),
    columns=["disease_id", "gene_id", "sentence_count", "doc_count"]
)


# # Calculate the Number of Occurences for Gene and Disease Separately

# In[ ]:


train_df = pd.read_csv("stratified_data/training_set.csv")
dev_df = pd.read_csv("stratified_data/dev_set.csv")
test_df = pd.read_csv("stratified_data/test_set.csv")


# In[ ]:


training_set = pd.merge(candidate_df, train_df, how='right', on=["disease_id", "gene_id"])
dev_set = pd.merge(candidate_df, dev_df, how='right', on=["disease_id", "gene_id"])
test_set = pd.merge(candidate_df, test_df, how='right', on=["disease_id", "gene_id"])
no_pubmed_df = training_set[training_set["sentence_count"].isnull()].append(dev_set[dev_set["sentence_count"].isnull()])
no_pubmed_df = no_pubmed_df.append(test_set[test_set["sentence_count"].isnull()])


# In[ ]:


data = []
for row in tqdm.tqdm(no_pubmed_df[["disease_id", "gene_id"]].values):
    document_disease = candidate_df[candidate_df["disease_id"] == row[0]]["doc_num"].sum()
    document_gene = candidate_df[candidate_df["gene_id"] == row[1]]["doc_num"].sum()
    sentence_disease = candidate_df[candidate_df["disease_id"] == row[0]]["sentence_count"].sum()
    sentence_gene = candidate_df[candidate_df["gene_id"] == row[1]]["sentence_count"].sum()
    data.append([document_disease, document_gene, sentence_disease, sentence_gene])


# ## Write To File

# After calulating above, the last step is to write to a file and use the dataset in the entity prediction notebook.

# In[ ]:


no_pubmed_df = pd.concat(
    [
        no_pubmed_df[["disease_id", "gene_id"]],
        pd.DataFrame(
            data,
            index=no_pubmed_df.index,
            columns=["disease_doc_count", "gene_doc_count", "disease_sen_count", "gene_sen_count"]
        )
    ], axis=1
)
no_pubmed_df["p_value"] = 1
no_pubmed_df.to_csv("disease_gene_npubmed_summary_stats.csv", index=False)

