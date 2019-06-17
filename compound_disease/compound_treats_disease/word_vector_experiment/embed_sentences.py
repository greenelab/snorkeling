
# coding: utf-8

# # Embed Compound Treats Disease Sentences

# This notebook is designed to embed compound treats disease (CtD) sentences. After word vectors have been trained, we embed sentences using the following steps:
# 
# 1. Load the total vocab generated from trained word vectors.
# 2. Cycle through each sentence
# 3. For each word in the sentence determine if word is in vocab
# 4. if yes assign index of no assign index for unknown token

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


# # Embed all of Compound Treats Disease Sentences

# This section embeds all candidate sentences. For each sentence, we place tags around each mention, tokenized the sentence and then matched each token to their corresponding word index. Any words missing from our vocab receive a index of 1. Lastly, the embedded sentences are exported as a sparse dataframe.

# In[17]:


word_dict_df = pd.read_table("results/compound_treats_disease_word_dict.tsv")
word_dict = {word[0]:word[1] for word in word_dict_df.values.tolist()}


# In[18]:


limit = 1000000
total_candidate_count = total_candidates_df.shape[0]

for offset in list(range(0, total_candidate_count, limit)):
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
        .offset(offset)
        .limit(limit)
        .all()
    )
    
    max_length = total_candidates_df.sen_length.max()
    
    # if first iteration create the file
    if offset == 0:
        (
            generate_embedded_df(candidates, word_dict, max_length=max_length)
            .to_sparse()
            .to_csv(
                "results/all_embedded_cd_sentences.tsv",
                index=False, 
                sep="\t", 
                mode="w"
            )
        )
        
    # else append don't overwrite
    else:
        (
            generate_embedded_df(candidates, word_dict, max_length=max_length)
            .to_sparse()
            .to_csv(
                "results/all_embedded_cd_sentences.tsv",
                index=False, 
                sep="\t", 
                mode="a",
                header=False
            )
        )


# In[ ]:


os.system("cd results; xz all_embedded_cd_sentences.tsv")

