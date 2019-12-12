#!/usr/bin/env python
# coding: utf-8

# # Generate Word Vectors For Compound Treats Disease Sentences

# This notebook is designed to generate word vectors for compound treats disease (CtD) sentences. Using facebooks's fasttext, we trained word vectors using all sentences that contain a disease and gene mention. The model was trained using the following specifications:
# 
# | Parameter | Value |
# | --- | --- |
# | Size | 300 |
# | alpha | 0.005 | 
# | window | 2 |
# | epochs | 50 |
# | seed | 100 | 

# # Set Up Environment

# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import os
import pickle
import sys

sys.path.append(os.path.abspath('../../../modules'))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook

from gensim.models import FastText
from gensim.models import KeyedVectors

from utils.notebook_utils.dataframe_helper import load_candidate_dataframes, generate_embedded_df


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


from snorkel.learning.pytorch.rnn.rnn_base import mark_sentence
from snorkel.learning.pytorch.rnn.utils import candidate_to_tokens
from snorkel.models import Candidate, candidate_subclass


# In[5]:


CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])


# # Compound Treats Disease

# This section loads the dataframe that contains all compound treats disease candidate sentences and their respective dataset assignments.

# In[9]:


cutoff = 300
total_candidates_df = (
    pd
    .read_table("../dataset_statistics/results/all_ctd_map.tsv.xz")
    .query("sen_length < 300")
)
total_candidates_df.head(2)


# # Train Word Vectors

# This section trains the word vectors using the specifications described above.

# In[10]:


words_to_embed = []
candidates = (
    session
    .query(CompoundDisease)
    .filter(
        CompoundDisease.id.in_(
            total_candidates_df
            .candidate_id
            .astype(int)
            .tolist()
        )
    )
    .all()
)


# In[11]:


for cand in tqdm_notebook(candidates):
    args = [
                (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                (cand[1].get_word_start(), cand[1].get_word_end(), 2)
    ]
    words_to_embed.append(mark_sentence(candidate_to_tokens(cand), args))


# In[12]:


model = FastText(
    words_to_embed, 
    window=2, 
    negative=10, 
    iter=50, 
    sg=1, 
    workers=4, 
    alpha=0.005, 
    size=300,
    seed=100
)


# In[13]:


(
    model
    .wv
    .save_word2vec_format(
        "results/compound_treats_disease_word_vectors.bin", 
        fvocab="results/compound_treats_disease_word_vocab.txt", 
        binary=False
        )
)


# In[14]:


model.wv.most_similar("diabetes")


# In[15]:


word_dict = {val[1]:val[0] for val in list(enumerate(model.wv.vocab.keys()))}
word_dict_df = (
    pd
    .DataFrame
    .from_dict(word_dict, orient="index")
    .reset_index()
    .rename({"index":"word", 0:"index"}, axis=1)
)
word_dict_df.to_csv("results/compound_treats_disease_word_dict.tsv", sep="\t", index=False)
word_dict_df.head(2)

