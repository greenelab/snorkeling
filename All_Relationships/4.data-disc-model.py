
# coding: utf-8

# # Train the Discriminator for Candidate Classification

# This notebook is designed to train ML algorithms: Long Short Term Memory Neural Net (LSTM) and SparseLogisticRegression (SLR) for candidate classification. 

# ## MUST RUN AT THE START OF EVERYTHING

# Set up the database for data extraction and load the Candidate subclass for the algorithms below

# In[ ]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import csv
import os

import numpy as np
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


from snorkel.annotations import FeatureAnnotator, LabelAnnotator, load_marginals
from snorkel.learning import SparseLogisticRegression
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.utils import RandomSearch
from snorkel.models import Candidate, FeatureKey, candidate_subclass


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

# This code will automatically load our labels and features that were generated in the [previous notebook](2.data-labeler.ipynb). 

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'labeler = LabelAnnotator(lfs=[])\n\n#L_train = labeler.load_matrix(session,split=0)\nL_dev = labeler.load_matrix(session,split=1)')


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

# In[ ]:


get_ipython().magic(u'time train_marginals = load_marginals(session, split=0)')
np.savetxt("pmacs/train_marginals", train_marginals)


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'"""\ntrain_kwargs = {\n    \'lr\':         0.001,\n    \'dim\':        100,\n    \'n_epochs\':   10,\n    \'dropout\':    0.5,\n    \'print_freq\': 1,\n    \'max_sentence_length\': 1000,\n}\n"""\nlstm = reRNN(seed=100, n_threads=4)\n#lstm.train(train_cands, train_marginals[0:10], X_dev=dev_cands, Y_dev=L_dev[0:10], **train_kwargs)')


# ### Write the Training data to an External File

# In[ ]:


import csv
chunksize = 100000
start = 0
with open('pmacs/dev_candidates_ends.csv', 'wb') as g, open("pmacs/dev_candidates_offsets.csv", "wb") as f:
    while True:
        train_cands = (
                session
                .query(DiseaseGene)
                .filter(DiseaseGene.split == 0)
                .order_by(DiseaseGene.id)
                .limit(chunksize)
                .offset(start)
                .all()
        )
        
        if not train_cands:
            break

        output = csv.writer(f)
        for c in tqdm.tqdm(train_cands):
            data, ends = lstm._preprocess_data([c], extend=True)
            output.writerow(data[0])
            g.write("{}\n".format(ends[0]))

        start += chunksize


# ### Save the word dictionary to an External File

# In[ ]:


import csv
with open("pmacs/train_word_dict.csv", 'w') as f:
    output = csv.DictWriter(f, fieldnames=["Key", "Value"])
    output.writeheader()
    for key in tqdm.tqdm(lstm.word_dict.d):
        output.writerow({'Key':key, 'Value': lstm.word_dict.d[key]})


# ### Save the Development Candidates to an External File

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'dev_cands = (\n        session\n        .query(DiseaseGene)\n        .filter(DiseaseGene.split == 1)\n        .order_by(DiseaseGene.id)\n        .all()\n)')


# In[ ]:


import csv
with open('pmacs/dev_candidates_ends.csv', 'wb') as g, open("pmacs/dev_candidates_offsets.csv", "wb") as f:
    output = csv.writer(f)
    for c in tqdm.tqdm(dev_cands):
        data, ends = lstm._preprocess_data([c])
        output.writerow(data[0])
        g.write("{}\n".format(ends[0]))

