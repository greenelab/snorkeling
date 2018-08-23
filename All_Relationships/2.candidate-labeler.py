
# coding: utf-8

# # Label The Candidates!

# This notebook corresponds to labeling and genearting features for each extracted candidate from the [previous notebook](1.data-loader.ipynb).

# ## MUST RUN AT THE START OF EVERYTHING

# Load all the imports and set up the database for database operations. Plus, set up the particular candidate type this notebook is going to work with. 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict, OrderedDict
import csv
import os
import re


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook


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


edge_type = "cg"
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


train_candidate_df = pd.read_excel("data/compound_gene/sentence_labels.xlsx")
train_candidate_df.head(2)


# In[ ]:


train_candidate_ids = list(map(int, train_candidate_df.candidate_id.values))[0:10]


# In[ ]:


candidates = session.query(CompoundGene).filter(CompoundGene.id.in_(train_candidate_ids)).limit(100)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:


sv


# In[ ]:


c = sv.get_selected()
c


# # Label Functions

# Here is one of the fundamental part of this project. Below are the label functions that are used to give a candidate a label of 1,0 or -1 which corresponds to correct label, unknown label and incorrection label. The goal here is to develop functions that can label accurately label as many candidates as possible. This idea comes from the [data programming paradigm](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly), where the goal is to be able to create labels that machine learning algorithms can use for accurate classification.  

# In[ ]:


from utils.disease_gene_lf import DG_LFS
from utils.compound_gene_lf import CG_LFS
#from utils.gene_gene_lf import *
#from utils.compound_disease_lf import *


# # Label The Candidates

# Label each candidate based on the provided labels above. This code runs with realtive ease, but optimization is definitely needed when the number of label functions increases linearly.

# In[ ]:


from  sqlalchemy.sql.expression import func
labeler = LabelAnnotator(lfs=list(CG_LFS["CG_DB"].values()) + 
                         list(CG_LFS["CG_TEXT"].values()) +  
                         list(CG_LFS["CG_BICLUSTER"]) + 
                         list(DG_LFS["DG_TEXT"].values()))


# In[ ]:


def make_sentence_df(candidates):
    rows = list()
    for c in tqdm_notebook(candidates):
        row = OrderedDict()
        row['candidate_id'] = c.id
        row['compound'] = c[0].get_span()
        row['disease'] = c[1].get_span()
        row['drugbank_id'] = c.Compound_cid
        row['entrez_gene_id'] = c.Gene_cid
        row['sentence'] = c.get_parent().text
        rows.append(row)
    return pd.DataFrame(rows)


# ### Train Set

# In[ ]:


sql = '''
SELECT id from candidate
WHERE split = 9 and type='compound_gene'
ORDER BY RANDOM()
LIMIT 50000;
'''
target_cids = [x[0] for x in session.execute(sql)]


# In[ ]:


candidates = session.query(CompoundGene).filter(CompoundGene.id.in_(target_cids)).all()


# In[ ]:


train_df = make_sentence_df(candidates)
train_df.head(2)


# In[ ]:


writer = pd.ExcelWriter('data/compound_gene/sentence_labels_train.xlsx')
(train_df
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()


# ### Label Train Set

# In[ ]:


sql = '''
SELECT id from candidate
WHERE split = 9 and type='compound_gene'
ORDER BY RANDOM()
LIMIT 1000;
'''
target_cids = [x[0] for x in session.execute(sql)]


# In[ ]:


candidates = session.query(CompoundGene).filter(CompoundGene.id.in_(target_cids)).all()


# In[ ]:


train_hand_df = make_sentence_df(candidates)
train_hand_df.head(2)


# In[ ]:


writer = pd.ExcelWriter('data/compound_gene/sentence_labels_train_dev.xlsx')
(train_hand_df
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()


# ### Dev Set

# In[ ]:


sql = '''
SELECT id from candidate
WHERE split = 10 and type='compound_gene'
ORDER BY RANDOM()
LIMIT 10000;
'''
gold_cids = [x[0] for x in session.execute(sql)]
gold_cids


# In[ ]:


candidates = session.query(CompoundGene).filter(CompoundGene.id.in_(gold_cids)).all()


# In[ ]:


dev_df = make_sentence_df(candidates)
dev_df.head(2)


# In[ ]:


writer = pd.ExcelWriter('data/compound_gene/sentence_labels_dev.xlsx')
(dev_df
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()


# # Quickly Relabel Candidates

# Use this block here to re-label candidates that have already been labled from the above process.

# In[ ]:


train_df = pd.read_excel('data/compound_gene/sentence_labels.xlsx')
target_cids = train_df.candidate_id.astype(int).tolist()
len(target_cids)


# In[ ]:


cids = session.query(CompoundGene.id).filter(CompoundGene.id.in_(target_cids))
get_ipython().run_line_magic('time', 'L_train = labeler.apply(split=6, cids_query=cids, parallelism=5)')


# In[ ]:


dev_df = pd.read_excel("data/compound_gene/sentence_labels_dev.xlsx")
dev_df = dev_df[dev_df.curated_dsh.notnull()]
gold_cids = list(map(int, dev_df.candidate_id.values))
#gold_cids = np.loadtxt('data/compound_gene/labeled_dev_candidates.txt').astype(int).tolist()
len(gold_cids)


# In[ ]:


cids = session.query(Candidate.id).filter(Candidate.id.in_(gold_cids))
get_ipython().run_line_magic('time', 'L_dev = labeler.apply_existing(cids_query=cids, parallelism=5, clear=False)')


# In[ ]:


train_hand_df = pd.read_excel("data/compound_gene/sentence_labels_train_hand.xlsx")
train_hand_cids = train_hand_df[train_hand_df.curated_dsh.notnull()].candidate_id.astype(int).tolist()
len(train_hand_cids)


# In[ ]:


cids = session.query(Candidate.id).filter(Candidate.id.in_(train_hand_cids))
get_ipython().run_line_magic('time', 'L_train_hand_labeled = labeler.apply_existing(cids_query=cids, parallelism=5, clear=False)')

