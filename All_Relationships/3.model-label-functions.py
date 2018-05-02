
# coding: utf-8

# # Train the Generative Model for Accurate Labeling

# This notebook is designed to run the generative model snorkel uses for estimating the probability of each candidate being a true candidate (label of 1). 

# ## MUST RUN AT THE START OF EVERYTHING

# Import the necessary modules and set up the database for database operations.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter, OrderedDict, defaultdict
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc


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


from snorkel import SnorkelSession
from snorkel.annotations import FeatureAnnotator, LabelAnnotator, save_marginals
from snorkel.learning import GenerativeModel
from snorkel.learning.utils import MentionScorer
from snorkel.models import Candidate, FeatureKey, candidate_subclass, Label
from snorkel.utils import get_as_dict
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator


# In[4]:


edge_type = "dg"


# In[5]:


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
#L_gold_train = load_gold_labels(session, annotator_name='danich1', split=0)
#annotated_cands_train_ids = list(map(lambda x: L_gold_train.row_index[x], L_gold_train.nonzero()[0]))

sql = '''
SELECT candidate_id FROM gold_label
'''
gold_cids = [x[0] for x in session.execute(sql)]
cids = session.query(Candidate.id).filter(Candidate.id.in_(gold_cids))

L_gold_dev = load_gold_labels(session, annotator_name='danich1', cids_query=cids)
annotated_cands_dev_ids = list(map(lambda x: L_gold_dev.row_index[x], L_gold_dev.nonzero()[0]))


# In[ ]:


L_gold_dev


# In[16]:


train_candidate_ids = np.loadtxt('data/labeled_candidates.txt').astype(int).tolist()
train_candidate_ids


# In[10]:


dev_candidate_ids = np.loadtxt('data/labeled_dev_candidates.txt').astype(int).tolist()
dev_candidate_ids


# In[11]:


get_ipython().run_cell_magic('time', '', 'labeler = LabelAnnotator(lfs=[])\n\n# Only grab candidates that have human labels\ncids = session.query(Candidate.id).filter(Candidate.id.in_(train_candidate_ids))\nL_train = labeler.load_matrix(session, split=0) #\n\ncids = session.query(Candidate.id).filter(Candidate.id.in_(dev_candidate_ids))\nL_dev = labeler.load_matrix(session,cids_query=cids)')


# In[12]:


print("Total Data Shape:")
print(L_train.shape)


# In[13]:


L_train = L_train[np.unique(L_train.nonzero()[0]), :]
print("Total Data Shape:")
print(L_train.shape)


# In[14]:


L_dev.shape


# # Train the Generative Model

# Here is the first step of classification step of this project, where we train a gnerative model to discriminate the correct label each candidate will receive. Snorkel's generative model uses a Gibbs Sampling on a [factor graph](http://deepdive.stanford.edu/assets/factor_graph.pdf), to generate the probability of a potential candidate being a true candidate (label of 1).

# In[17]:


get_ipython().run_cell_magic('time', '', 'from snorkel.learning import GenerativeModel\n\ngen_model = GenerativeModel()\ngen_model.train(\n    L_train,\n    epochs=30,\n    decay=0.95,\n    step_size=0.1 / L_train.shape[0],\n    reg_param=1e-6,\n    threads=50,\n    verbose=True\n)')


# In[18]:


gen_model.weights.lf_accuracy


# In[19]:


from utils.disease_gene_lf import LFS
learned_stats_df = gen_model.learned_lf_stats()
learned_stats_df.index = list(LFS)
learned_stats_df


# In[20]:


get_ipython().run_line_magic('time', 'train_marginals = gen_model.marginals(L_train)')


# In[ ]:


print(len(train_marginals[train_marginals > 0.5]))


# In[ ]:


plt.hist(train_marginals, bins=20)
plt.title("Training Marginals for Gibbs Sampler")
plt.show()


# ## ROC of Generative Model

# In[21]:


dev_marginals = gen_model.marginals(L_dev)


# In[ ]:


fpr, tpr, threshold = roc_curve(L_gold_dev.todense(), dev_marginals)
plt.plot([0,1], [0,1])
plt.plot(fpr, tpr, label='AUC {:.2f}'.format(auc(fpr, tpr)))
plt.legend()


# In[ ]:


L_dev.lf_stats(session, L_gold_dev[L_gold_dev!=0].T, gen_model.learned_lf_stats()['Accuracy'])


# ## Individual Candidate Error Analysis

# In[ ]:


tp, fp, tn, fn = gen_model.error_analysis(session, L_dev, L_gold_dev)


# In[ ]:


from snorkel.viewer import SentenceNgramViewer

# NOTE: This if-then statement is only to avoid opening the viewer during automated testing of this notebook
# You should ignore this!
import os
if 'CI' not in os.environ:
    sv = SentenceNgramViewer(fp, session)
else:
    sv = None


# In[ ]:


sv


# In[ ]:


c = sv.get_selected() if sv else list(fp.union(fn))[0]
c


# In[ ]:


c.labels


# ## Generate Excel File of Train Data

# In[22]:


pair_df = pd.read_csv("data/disease-gene-pairs-association.csv.xz", compression='xz')
pair_df.head(2)


# In[23]:


rows = list()
for i in tqdm.tqdm(range(L_dev.shape[0])):
    row = OrderedDict()
    candidate = L_dev.get_candidate(session, i)
    row['candidate_id'] = candidate.id
    row['disease'] = candidate[0].get_span()
    row['gene'] = candidate[1].get_span()
    row['doid_id'] = candidate.Disease_cid
    row['entrez_gene_id'] = candidate.Gene_cid
    row['sentence'] = candidate.get_parent().text
    row['label'] = train_marginals[i]
    rows.append(row)
sentence_df = pd.DataFrame(rows)
sentence_df['entrez_gene_id'] = sentence_df.entrez_gene_id.astype(int)
sentence_df.head(2)


# In[24]:


sentence_df = pd.merge(
    sentence_df,
    pair_df[["doid_id", "entrez_gene_id", "doid_name", "gene_symbol"]],
    on=["doid_id", "entrez_gene_id"],
    how="left"
)
sentence_df.head(2)


# In[25]:


sentence_df = pd.concat([
    sentence_df,
    pd.DataFrame(L_dev.todense(), columns=list(LFS))
], axis='columns')

sentence_df.tail()


# In[26]:


writer = pd.ExcelWriter('data/sentence-labels-dev.xlsx')
(sentence_df
 .sort_values("label", ascending=False)
 .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()


# # Save Training Marginals

# Save the training marginals for [Notebook 4](4.data-disc-model).

# In[ ]:


np.savetxt("vanilla_lstm/lstm_disease_gene_holdout/subsampled/train_marginals_subsampled.txt", train_marginals)


# In[ ]:


#%time save_marginals(session, L_train, train_marginals)

