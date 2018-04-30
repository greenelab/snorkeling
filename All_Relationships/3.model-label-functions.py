
# coding: utf-8

# # Train the Generative Model for Accurate Labeling

# This notebook is designed to run the generative model snorkel uses for estimating the probability of each candidate being a true candidate (label of 1). 

# ## MUST RUN AT THE START OF EVERYTHING

# Import the necessary modules and set up the database for database operations.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter
from collections import defaultdict
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc


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


from snorkel import SnorkelSession
from snorkel.annotations import FeatureAnnotator, LabelAnnotator, save_marginals
from snorkel.learning import GenerativeModel
from snorkel.learning.utils import MentionScorer
from snorkel.models import Candidate, FeatureKey, candidate_subclass
from snorkel.utils import get_as_dict
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator


# In[ ]:


edge_type = "dg"


# In[ ]:


if edge_type == "dg":
    DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
elif edge_type == "gg":
    GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
elif edge_type == "cg":
    CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
elif edge_type == "cd":
    CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])
else:
    print("Please pick a valid edge type")


# # Load preprocessed data 

# This code will load the label matrix that was generated in the previous notebook ([Notebook 2](2.data-labeler.ipynb)). **Disclaimer**: this block might break, which means that the snorkel code is still using its old code. The problem with the old code is that sqlalchemy will attempt to load all the labels into memory. Doesn't sound bad if you keep the amount of labels small, but doesn't scale when the amount of labels increases exponentially. Good news is that there is a pull request to fix this issue. [Check it out here!](https://github.com/HazyResearch/snorkel/pull/789)

# In[ ]:


from snorkel.annotations import load_gold_labels
L_gold_train = load_gold_labels(session, annotator_name='danich1', split=0)
annotated_cands_train_ids = list(map(lambda x: L_gold_train.row_index[x], L_gold_train.nonzero()[0]))

L_gold_dev = load_gold_labels(session, annotator_name='danich1', split=1)
annotated_cands_dev_ids = list(map(lambda x: L_gold_dev.row_index[x], L_gold_dev.nonzero()[0]))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'labeler = LabelAnnotator(lfs=[])\n\n# Only grab candidates that have human labels\ncids = session.query(Candidate.id).filter(Candidate.id.in_(annotated_cands_train_ids))\nL_train = labeler.load_matrix(session,cids_query=cids)\n\ncids = session.query(Candidate.id).filter(Candidate.id.in_(annotated_cands_dev_ids))\nL_dev = labeler.load_matrix(session,cids_query=cids)')


# In[ ]:


print("Total Data Shape:")
print(L_train.shape)


# In[ ]:


L_train.get_candidate(session, 1)


# # Train the Generative Model

# Here is the first step of classification step of this project, where we train a gnerative model to discriminate the correct label each candidate will receive. Snorkel's generative model uses a Gibbs Sampling on a [factor graph](http://deepdive.stanford.edu/assets/factor_graph.pdf), to generate the probability of a potential candidate being a true candidate (label of 1).

# In[ ]:


from snorkel.learning import GenerativeModel

gen_model = GenerativeModel()
get_ipython().run_line_magic('time', 'gen_model.train(L_train, epochs=30, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6, threads=50, verbose=True)')


# In[ ]:


gen_model.weights.lf_accuracy


# In[ ]:


get_ipython().run_line_magic('time', 'train_marginals = gen_model.marginals(L_train)')


# In[ ]:


gen_model.learned_lf_stats()


# In[ ]:


print(len(train_marginals[train_marginals > 0.5]))


# In[ ]:


plt.hist(train_marginals, bins=20)
plt.title("Training Marginals for Gibbs Sampler")
plt.show()


# In[ ]:


tp, fp, tn, fn = gen_model.error_analysis(session, L_train, L_gold_train)


# In[ ]:


from snorkel.viewer import SentenceNgramViewer

# NOTE: This if-then statement is only to avoid opening the viewer during automated testing of this notebook
# You should ignore this!
import os
if 'CI' not in os.environ:
    sv = SentenceNgramViewer(fn, session)
else:
    sv = None


# In[ ]:


sv


# In[ ]:


c = sv.get_selected() if sv else list(fp.union(fn))[0]
c


# In[ ]:


c.labels


# In[ ]:


c.Gene_cid


# In[ ]:


L_train.lf_stats(session, L_gold_train[L_gold_train!=0].T, gen_model.learned_lf_stats()['Accuracy'])


# # Save Training Marginals

# Save the training marginals for [Notebook 4](4.data-disc-model).

# In[ ]:


np.savetxt("vanilla_lstm/lstm_disease_gene_holdout/subsampled/train_marginals_subsampled.txt", train_marginals)


# In[ ]:


#%time save_marginals(session, L_train, train_marginals)

