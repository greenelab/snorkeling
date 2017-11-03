
# coding: utf-8

# # Train the Discriminator for Candidate Classification on the Sentence Level

# This notebook is designed to train ML algorithms: Long Short Term Memory Neural Net (LSTM) and SparseLogisticRegression (SLR) for candidate classification. 

# ## MUST RUN AT THE START OF EVERYTHING

# Set up the database for data extraction and load the Candidate subclass for the algorithms below

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import csv
import os

import numpy as np
import pandas as pd
import tqdm


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


from snorkel.annotations import FeatureAnnotator, LabelAnnotator, load_marginals
from snorkel.learning import SparseLogisticRegression
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.utils import RandomSearch
from snorkel.models import Candidate, FeatureKey, candidate_subclass


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

# This code will automatically load our labels and features that were generated in the [previous notebook](2.data-labeler.ipynb). 

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'labeler = LabelAnnotator(lfs=[])\n\nL_train = labeler.load_matrix(session,split=0)\nL_dev = labeler.load_matrix(session,split=1)')


# In[ ]:


print "Total Data Shape:"
print L_train.shape
print L_dev.shape
print


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'featurizer = FeatureAnnotator()\n\nF_train = featurizer.load_matrix(session, split=0)\nF_dev = featurizer.load_matrix(session, split=1)')


# In[ ]:


print "Total Data Shape:"
print F_train.shape
print F_dev.shape
print


# # Train Sparse Logistic Regression Disc Model

# Here we train an SLR. To find the optimal hyperparameter settings this code uses a [random search](http://scikit-learn.org/stable/modules/grid_search.html) instead of iterating over all possible combinations of parameters. After the final model has been found, it is saved in the checkpoints folder to be loaded in the [next notebook](5.data-analysis.ipynb). Furthermore, the weights for the final model are output into a text file to be analyzed as well.

# In[ ]:


get_ipython().magic(u'time train_marginals = load_marginals(session, split=0)')


# In[ ]:


# Searching over learning rate
param_ranges = {
    'lr' : [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'l1_penalty' : [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
    'l2_penalty' : [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
}
model_hyperparams = {
    'n_epochs' : 50,
    'rebalance' : 0.5,
    'print_freq' : 25
}
searcher = RandomSearch(SparseLogisticRegression, param_ranges, F_train,
                        Y_train=train_marginals, n=5, model_hyperparams=model_hyperparams)


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'np.random.seed(100)\ndisc_model, run_stats = searcher.fit(F_dev, L_dev, n_threads=4)')


# In[ ]:


w, b = disc_model.get_weights()


# In[ ]:


# Write the weights and features for further processing
annot_select_query = FeatureKey.__table__.select().order_by(FeatureKey.id)
with open("LR_model.csv", "w") as f:
    fieldnames = ["Weight", "Feature"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for weight, feature in tqdm.tqdm(zip(w, session.execute(annot_select_query))):
        writer.writerow({"Weight": weight, "Feature":feature[1]})


# ## Train a LSTM Disc Model

# This block of code trains an LSTM. An LSTM is a special type of recurrent nerual network that retains a memory of past values over period of time. ([Further explaination here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)). The problem with the code below is that sqlalchemy runs into an out of memory error on my computer during the preprocessing step. As a consequence we have to resort loading this data onto University of Pennsylvania's Performance Computing Cluster. The data that gets preprocessed is exported to a text file and then get shipped towards the cluster.

# In[6]:


directory = 'stratified_data/lstm_disease_gene_holdout/'


# In[9]:


get_ipython().magic(u'time train_marginals = load_marginals(session, split=0)')
np.savetxt("{}/train_marginals".format(directory), train_marginals)


# In[7]:


get_ipython().run_cell_magic(u'time', u'', u'"""\ntrain_kwargs = {\n    \'lr\':         0.001,\n    \'dim\':        100,\n    \'n_epochs\':   10,\n    \'dropout\':    0.5,\n    \'print_freq\': 1,\n    \'max_sentence_length\': 1000,\n}\n"""\nlstm = reRNN(seed=100, n_threads=4)\n#lstm.train(train_cands, train_marginals[0:10], X_dev=dev_cands, Y_dev=L_dev[0:10], **train_kwargs)')


# ### Write the Training data to an External File

# In[9]:


get_ipython().run_cell_magic(u'time', u'', u'field_names = ["disease_id", "disease_char_start", "disease_char_end", "gene_id", "gene_char_start", "gene_char_end", "sentence", "pubmed"]\nchunksize = 100000\nstart = 0\n\nwith open(\'{}/train_candidates_ends.csv\'.format(directory), \'wb\') as g:\n    with open("{}/train_candidates_offsets.csv".format(directory), "wb") as f:\n        with open("{}/train_candidates_sentences.csv".format(directory), "wb") as h:\n            output = csv.writer(f)\n            writer = csv.DictWriter(h, fieldnames=field_names)\n            writer.writeheader()\n\n            while True:\n                train_cands = (\n                        session\n                        .query(DiseaseGene)\n                        .filter(DiseaseGene.split == 0)\n                        .order_by(DiseaseGene.id)\n                        .limit(chunksize)\n                        .offset(start)\n                        .all()\n                )\n\n                if not train_cands:\n                    break\n\n                \n                for c in tqdm.tqdm(train_cands):\n                    data, ends = lstm._preprocess_data([c], extend=True)\n                    output.writerow(data[0])\n                    g.write("{}\\n".format(ends[0]))\n                    \n                    row = {\n                        "disease_id": c.Disease_cid, "disease_char_start":c[0].char_start, \n                        "disease_char_end": c[0].char_end, "gene_id": c.Gene_cid, \n                        "gene_char_start":c[1].char_start, "gene_char_end":c[1].char_end, \n                        "sentence": c.get_parent().text, "pubmed", c.get_parent().get_parent().name\n                        }\n                \n                    writer.writerow(row)\n\n                start += chunksize')


# ### Save the word dictionary to an External File

# In[10]:


get_ipython().run_cell_magic(u'time', u'', u'with open("{}/train_word_dict.csv".format(directory), \'w\') as f:\n    output = csv.DictWriter(f, fieldnames=["Key", "Value"])\n    output.writeheader()\n    for key in tqdm.tqdm(lstm.word_dict.d):\n        output.writerow({\'Key\':key, \'Value\': lstm.word_dict.d[key]})')


# ### Save the Development Candidates to an External File

# In[11]:


dev_cands = (
        session
        .query(DiseaseGene)
        .filter(DiseaseGene.split == 1)
        .order_by(DiseaseGene.id)
        .all()
)

dev_cand_labels = pd.read_csv("stratified_data/dev_set.csv")
hetnet_set = set(map(tuple,dev_cand_labels[dev_cand_labels["hetnet"] == 1][["disease_ontology", "gene_id"]].values))


# In[12]:


get_ipython().run_cell_magic(u'time', u'', u'field_names = ["disease_id", "disease_char_start", "disease_char_end", "gene_id", "gene_char_start", "gene_char_end", "sentence", "pubmed"]\nwith open(\'{}/dev_candidates_offset.csv\'.format(directory), \'wb\') as g:\n    with open(\'{}/dev_candidates_labels.csv\'.format(directory), \'wb\') as f:\n        with open(\'{}/dev_candidates_sentences.csv\'.format(directory), \'wb\') as h:\n            \n            output = csv.writer(g)\n            label_output = csv.writer(f)\n            writer = csv.DictWriter(h, fieldnames=field_names)\n            writer.writeheader()\n            \n            for c in tqdm.tqdm(dev_cands):\n                data, ends = lstm._preprocess_data([c])\n                output.writerow(data[0])\n                label_output.writerow([1 if (c.Disease_cid, int(c.Gene_cid)) in hetnet_set else -1])\n                \n                row = {\n                "disease_id": c.Disease_cid, "disease_char_start":c[0].char_start, \n                "disease_char_end": c[0].char_end, "gene_id": c.Gene_cid, \n                "gene_char_start":c[1].char_start, "gene_char_end":c[1].char_end, \n                "sentence": c.get_parent().text, "pubmed", c.get_parent().get_parent().name\n                }\n                \n                writer.writerow(row) ')


# ### Save the Test Candidates to an External File

# In[13]:


test_cands = (
        session
        .query(DiseaseGene)
        .filter(DiseaseGene.split == 2)
        .order_by(DiseaseGene.id)
        .all()
)

dev_cand_labels = pd.read_csv("stratified_data/test_set.csv")
hetnet_set = set(map(tuple,dev_cand_labels[dev_cand_labels["hetnet"] == 1][["disease_ontology", "gene_id"]].values))


# In[14]:


get_ipython().run_cell_magic(u'time', u'', u'field_names = ["disease_id", "disease_char_start", "disease_char_end", "gene_id", "gene_char_start", "gene_char_end", "sentence", "pubmed"]\nwith open(\'{}/test_candidates_offset.csv\'.format(directory), \'wb\') as g:\n    with open(\'{}/test_candidates_labels.csv\'.format(directory), \'wb\') as f:\n        with open(\'{}/test_candidates_sentences.csv\'.format(directory), \'wb\') as h:\n            \n            output = csv.writer(g)\n            label_output = csv.writer(f)\n            writer = csv.DictWriter(h, fieldnames=field_names)\n            writer.writeheader()\n            \n            for c in tqdm.tqdm(test_cands):\n                data, ends = lstm._preprocess_data([c])\n                output.writerow(data[0])\n                label_output.writerow([1 if (c.Disease_cid, int(c.Gene_cid)) in hetnet_set else -1])\n                \n                row = {\n                "disease_id": c.Disease_cid, "disease_char_start":c[0].char_start, \n                "disease_char_end": c[0].char_end, "gene_id": c.Gene_cid, \n                "gene_char_start":c[1].char_start, "gene_char_end":c[1].char_end, \n                "sentence": c.get_parent().text, "pubmed", c.get_parent().get_parent().name\n                }\n                \n                writer.writerow(row) ')

