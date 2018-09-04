
# coding: utf-8

# # Label The Candidates!

# This notebook corresponds to labeling each extracted candidate from the [previous notebook](1.data-loader.ipynb).

# ## MUST RUN AT THE START OF EVERYTHING

# Load all the imports and set up the database for database operations. Plus, set up the particular candidate type this notebook is going to work with. 

# In[ ]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import os

import pandas as pd

# This module is designed to help create dataframes for each candidate sentence
from utils.notebook_utils.dataframe_helper import make_sentence_df, write_candidates_to_excel


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


from snorkel.annotations import LabelAnnotator
from snorkel.models import candidate_subclass
from snorkel.models import Candidate
from snorkel.viewer import SentenceNgramViewer


# In[ ]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])


# ## Write the Candidates to Excel File

# In[ ]:


sql_statements = [
    '''
    SELECT id from candidate
    WHERE split = 9 and type='compound_disease'
    ORDER BY RANDOM()
    LIMIT 50;
    ''',
    
    '''
    SELECT id from candidate
    WHERE split = 9 and type='compound_disease'
    ORDER BY RANDOM()
    LIMIT 10;
    ''',

    '''
    SELECT id from candidate
    WHERE split = 10 and type='compound_disease'
    ORDER BY RANDOM()
    LIMIT 10;
    '''
]

spreadsheet_names = {
    'train': 'data/compound_disease/sentence_labels_train.xlsx',
    'train_hand_label': 'data/compound_disease/sentence_labels_train_dev.xlsx',
    'dev': 'data/compound_disease/sentence_labels_dev.xlsx'
}


# In[ ]:


for sql, spreadsheet_name in zip(sql_statements, spreadsheet_names.values()):
    target_cids = [x[0] for x in session.execute(sql)]
    candidates = (
        session
        .query(CandidateClass)
        .filter(CandidateClass.id.in_(target_cids))
        .all()
    )
    candidate_df = make_sentence_df(candidates)
    write_candidates_to_excel(candidate_df, spreadsheet_name)


# # Develop Label Functions

# ## Look at potential Candidates

# Use this to look at loaded candidates from a given set. The constants represent the index to retrieve the appropiate set. Ideally, here is where one can look at a subset of the candidate and develop label functions for candidate labeling.

# In[ ]:


train_df = pd.read_excel(spreadsheet_names['train'])
train_df.head(2)


# In[ ]:


train_candidate_ids = train_df.candidate_id.astype(int)


# In[ ]:


candidates = (
    session
    .query(CompoundGene)
    .filter(CompoundGene.id.in_(train_candidate_ids))
    .limit(10)
    .offset(0)
)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:


sv


# In[ ]:


c = sv.get_selected()
c


# ## Bicluster Dataframe formation

# In[ ]:


url = "https://zenodo.org/record/1035500/files/"
dep_path = "part-ii-dependency-paths-chemical-disease-sorted-with-themes.txt"
file_dist = "part-i-chemical-disease-path-theme-distributions.txt"
output_file = "data/compound_disease/biclustering/compound_disease_bicluster_results.tsv"


# In[ ]:


from utils.notebook_utils.bicluster import create_bicluster_df
create_bicluster_df(url+dep_path, url+file_dist, output_file)


# # Label Functions

# Here is one of the fundamental part of this project. Below are the label functions that are used to give a candidate a label of 1,0 or -1 which corresponds to correct label, unknown label and incorrection label. The goal here is to develop functions that can label accurately label as many candidates as possible. This idea comes from the [data programming paradigm](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly), where the goal is to be able to create labels that machine learning algorithms can use for accurate classification.  

# In[ ]:


from utils.label_functions.disease_gene_lf import DG_LFS
from utils.label_functions.compound_gene_lf import CG_LFS
from utils.label_functions.compound_disease_lf import CD_LFS
#from utils.gene_gene_lf import GG_LFS


# # Label The Candidates

# Label each candidate based on the provided labels above. This code runs with realtive ease, but optimization is definitely needed when the number of label functions increases linearly.

# In[ ]:


label_functions = list(CG_LFS["CbG_DB"].values()) + 
                  list(CG_LFS["CbG_TEXT"].values()) +   
                  list(DG_LFS["DaG_TEXT"].values())

labeler = LabelAnnotator(lfs=label_functions)


# # Quickly Relabel Candidates

# Use this block here to re-label candidates that have already been labled from the above process.

# In[ ]:


train_df = pd.read_excel(spreadsheet_names['train'])
train_cids = train_df.candidate_id.astype(int).tolist()
train_df.head(2)


# In[ ]:


dev_df = pd.read_excel(spreadsheet_names['dev'])
dev_df = dev_df[dev_df.curated_dsh.notnull()]
dev_cids = list(map(int, dev_df.candidate_id.values))
dev_df.head(2)


# In[ ]:


train_hand_df = pd.read_excel(spreadsheet_names['train_hand_label'])
train_hand_cids = train_hand_df[train_hand_df.curated_dsh.notnull()].candidate_id.astype(int).tolist()
train_hand_df.head(2)


# In[ ]:


for cid_list in [train_cids, train_hand_cids, dev_cids]:
    cids = session.query(CompoundGene.id).filter(CompoundGene.id.in_(cid_list))
    get_ipython().magic(u'time labeler.apply(cids_query=cids, parallelism=5)')

