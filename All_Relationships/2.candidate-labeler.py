
# coding: utf-8

# # Label The Candidates! Extract The Features!

# This notebook corresponds to labeling and genearting features for each extracted candidate from the [previous notebook](1.data-loader.ipynb).

# ## MUST RUN AT THE START OF EVERYTHING

# Load all the imports and set up the database for database operations. Plus, set up the particular candidate type this notebook is going to work with. 

# In[ ]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

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
from snorkel.models import Candidate
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


# # Look at potential Candidates

# Use this to look at loaded candidates from a given set. The constants represent the index to retrieve the appropiate set. Ideally, here is where one can look at a subset of the candidate and develop label functions for candidate labeling.

# In[ ]:

TRAIN = 0
DEV = 1


# In[ ]:

candidates = session.query(DiseaseGene).filter(DiseaseGene.split==TRAIN).limit(100)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:

sv


# # Label Functions

# Here is one of the fundamental part of this project. Below are the label functions that are used to give a candidate a label of 1,0 or -1 which corresponds to correct label, unknown label and incorrection label. The goal here is to develop functions that can label accurately label as many candidates as possible. This idea comes from the [data programming paradigm](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly), where the goal is to be able to create labels that machine learning algorithms can use for accurate classification.  

# In[ ]:

if edge_type == "dg":
    from utils.disease_gene_lf import get_lfs
elif edge_type == "gg":
    from utils.gene_gene_lf import *
elif edge_type == "cg":
    from utils.compound_gene_lf import *
elif edge_type == "cd":
    from utils.compound_disease_lf import *
else:
    print("Please pick a valid edge type")


# In[ ]:

candidates = session.query(DiseaseGene).filter(DiseaseGene.split==0).limit(1).all()
LF_DEBUG(candidates[0])


# In[ ]:

LFs = get_lfs()


# # Label The Candidates

# Label each candidate based on the provided labels above. This code runs with realtive ease, but optimization is definitely needed when the number of label functions increases linearly.

# In[ ]:

labeler = LabelAnnotator(lfs=LFs)

cids = session.query(Candidate.id).filter(Candidate.split==0)
get_ipython().magic(u'time L_train = labeler.apply(split=0, cids_query=cids, parallelism=5)')

cids = session.query(Candidate.id).filter(Candidate.split==1)
get_ipython().magic(u'time L_dev = labeler.apply_existing(split=1, cids_query=cids, parallelism=5, clear=False)')

cids = session.query(Candidate.id).filter(Candidate.split==2)
get_ipython().magic(u'time L_test = labeler.apply_existing(split=2, cids_query=cids, parallelism=5, clear=False)')


# # Generate Candidate Features

# In conjunction with each candidate label, generate candidate features that will be used by some machine learning algorithms (notebook 4). This step is broken as insert takes an **incredibly** long time to run. Had to do roundabout way to load the features. **Do not run this block** and refer to the code block below. Gonna need to debug this part, when I get time.

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'featurizer = FeatureAnnotator()\nfeaturizer.apply(split=0, clear=False)\n\nF_dev = featurizer.apply_existing(split=1, parallelism=5, clear=False)\nF_test = featurizer.apply_existing(split=2, parallelism=5, clear=False)')


# # Work Around for above code

# As mentioned above this code is the workaround for the broken featurizer. The intuition behind this section is to write all the generated features to a sql text file. Exploting the psql's COPY command, the time taken for inserting features drops to ~30 minutues (compared to 1 week+).

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'\ngroup = 0\nchunksize = 1e5\nseen = set()\nfeature_key_hash = defaultdict(int)\nfeat_counter = 0\n\nwith open(\'feature_key.sql\', \'wb\') as f:\n    with open(\'feature.sql\', \'wb\') as g:\n        # Write the headers\n        f.write("COPY feature_key(\\"group\\", name, id) from stdin with CSV DELIMITER \'\t\' QUOTE \'\\"\';\\n")\n        g.write("COPY feature(value, candidate_id, key_id) from stdin with CSV DELIMITER \'\t\' QUOTE \'\\"\';\\n")\n        \n        # Set up the writers\n        feature_key_writer = csv.writer(f, delimiter=\'\\t\',  quoting=csv.QUOTE_NONNUMERIC)\n        feature_writer = csv.writer(g, delimiter=\'\\t\', quoting=csv.QUOTE_NONNUMERIC)\n        \n        # For each split get and generate features\n        for split in [0,1,2]:\n    \n            #reset pointer to cycle through database again\n            pointer = 0\n            \n            print(split)\n            candidate_query = session.query(Candidate).filter(Candidate.split==split).limit(chunksize)\n            \n            while True:\n                candidates = candidate_query.offset(pointer).all()\n                \n                if not candidates:\n                    break\n\n                for c in tqdm.tqdm(candidates):\n                    try:\n                        for name, value in get_span_feats(c):\n\n                            # If the training set, set the feature hash\n                            if split == 0:\n                                if name not in feature_key_hash:\n                                    feature_key_hash[name] = feat_counter\n                                    feat_counter = feat_counter + 1\n                                    feature_key_writer.writerow([group, name, feature_key_hash[name]])\n\n                            if name in feature_key_hash:\n                                # prevent duplicates from being written to the file\n                                if (c.id, name) not in seen:\n                                    feature_writer.writerow([value, c.id, feature_key_hash[name]])\n                                    seen.add((c.id, name))\n\n                        #To prevent memory overload\n                        seen = set()\n                    \n                    except Exception as e:\n                        print(e.message)\n                        print(c)\n                        print(c.get_parent().text)\n\n                # update pointer for database\n                pointer = pointer + chunksize')


# # Generate Coverage Stats

# Before throwing our labels at a machine learning algorithm take a look at some quick stats. The code below will show the coverage and conflicts of each label function. Furthermore, this code will show the dimension of each label matrix.

# In[ ]:

print L_train.lf_stats(session, )


# In[ ]:

print L_dev.lf_stats(session, )

