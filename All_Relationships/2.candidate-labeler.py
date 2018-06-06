
# coding: utf-8

# # Label The Candidates!

# This notebook corresponds to labeling and genearting features for each extracted candidate from the [previous notebook](1.data-loader.ipynb).

# ## MUST RUN AT THE START OF EVERYTHING

# Load all the imports and set up the database for database operations. Plus, set up the particular candidate type this notebook is going to work with. 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import csv
import os
import re


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm


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


from snorkel.annotations import FeatureAnnotator, LabelAnnotator
from snorkel.features import get_span_feats
from snorkel.models import candidate_subclass
from snorkel.models import Candidate, GoldLabel
from snorkel.viewer import SentenceNgramViewer


# In[ ]:


edge_type = "dg"
debug = False


# In[ ]:


if edge_type == "dg":
    DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
    edge = "disease_gene"
elif edge_type == "gg":
    GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
    edge = "gene_gene"
elif edge_type == "cg":
    CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
    edge = "compound_gene"
elif edge_type == "cd":
    CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])
    edge = "compound_disease"
else:
    print("Please pick a valid edge type")


# # Develop Label Functions

# ## Look at potential Candidates

# Use this to look at loaded candidates from a given set. The constants represent the index to retrieve the appropiate set. Ideally, here is where one can look at a subset of the candidate and develop label functions for candidate labeling.

# In[ ]:


train_candidate_df = pd.read_excel("data/sentence-labels.xlsx")
train_candidate_df.head(2)


# In[ ]:


train_candidate_ids = list(map(int, train_candidate_df.candidate_id.values))[10:60]


# In[ ]:


candidates = session.query(DiseaseGene).filter(DiseaseGene.id.in_(train_candidate_ids)).limit(100)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:


sv


# In[ ]:


c = sv.get_selected()


# # Label Functions

# Here is one of the fundamental part of this project. Below are the label functions that are used to give a candidate a label of 1,0 or -1 which corresponds to correct label, unknown label and incorrection label. The goal here is to develop functions that can label accurately label as many candidates as possible. This idea comes from the [data programming paradigm](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly), where the goal is to be able to create labels that machine learning algorithms can use for accurate classification.  

# In[ ]:


if edge_type == "dg":
    from utils.disease_gene_lf import LFS, LF_DEBUG
elif edge_type == "gg":
    from utils.gene_gene_lf import *
elif edge_type == "cg":
    from utils.compound_gene_lf import *
elif edge_type == "cd":
    from utils.compound_disease_lf import *
else:
    print("Please pick a valid edge type")


# # Label The Candidates

# Label each candidate based on the provided labels above. This code runs with realtive ease, but optimization is definitely needed when the number of label functions increases linearly.

# In[ ]:


from  sqlalchemy.sql.expression import func
labeler = LabelAnnotator(lfs=list(LFS.values()))


# ### Train Set

# In[ ]:


sql = '''
SELECT id from candidate
WHERE split = 0 and type='disease_gene'
ORDER BY RANDOM()
LIMIT 50000;
'''
target_cids = [x[0] for x in session.execute(sql)]


# In[ ]:


target_cids


# In[ ]:


np.savetxt('data/labeled_candidates.txt', target_cids)


# ### Dev Set

# In[ ]:


sql = '''
SELECT candidate_id FROM gold_label
'''
gold_cids = [x[0] for x in session.execute(sql)]
gold_cids


# In[ ]:


sql = '''
SELECT id from candidate
WHERE split = 0 and type='disease_gene'
ORDER BY RANDOM()
LIMIT 10000;
'''
gold_cids = [x[0] for x in session.execute(sql)]
gold_cids


# In[ ]:


np.savetxt('data/labeled_dev_candidates.txt', gold_cids)


# # Quickly Relabel Candidates

# Use this block here to re-label candidates that have already been labled from the above process.

# In[ ]:


target_cids = np.loadtxt('data/labeled_candidates.txt').astype(int).tolist()


# In[ ]:


cids = session.query(DiseaseGene.id).filter(DiseaseGene.id.in_(target_cids))
get_ipython().run_line_magic('time', 'L_train = labeler.apply(split=0, cids_query=cids, parallelism=5)')


# In[ ]:


dev_df = pd.read_excel("data/sentence-labels-dev-hand-labeled.xlsx")
dev_df = dev_df[dev_df.curated_dsh.notnull()]
gold_cids = list(map(int, dev_df.candidate_id.values))
len(gold_cids)


# In[ ]:


cids = session.query(Candidate.id).filter(Candidate.id.in_(gold_cids))
get_ipython().run_line_magic('time', 'L_dev = labeler.apply_existing(cids_query=cids, parallelism=5, clear=False)')


# In[ ]:


sql = '''
SELECT candidate_id FROM gold_label
INNER JOIN Candidate ON Candidate.id=gold_label.candidate_id
WHERE Candidate.split=0;
'''
cids = session.query(Candidate.id).filter(Candidate.id.in_([x[0] for x in session.execute(sql)]))
get_ipython().run_line_magic('time', 'L_train_hand_labeled = labeler.apply_existing(cids_query=cids, parallelism=5, clear=False)')

