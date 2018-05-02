
# coding: utf-8

# # Train the Discriminator for Candidate Classification on the Sentence Level

# This notebook is designed to train ML algorithms: Long Short Term Memory Neural Net (LSTM) and SparseLogisticRegression (SLR) for candidate classification. 

# ## MUST RUN AT THE START OF EVERYTHING

# Set up the database for data extraction and load the Candidate subclass for the algorithms below

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import csv
import os

import numpy as np
import pandas as pd
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier


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
from snorkel.annotations import load_gold_labels
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


# ## Train a LSTM Disc Model

# This block of code trains an LSTM. An LSTM is a special type of recurrent nerual network that retains a memory of past values over period of time. ([Further explaination here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)). The problem with the code below is that sqlalchemy runs into an out of memory error on my computer during the preprocessing step. As a consequence we have to resort loading this data onto University of Pennsylvania's Performance Computing Cluster. The data that gets preprocessed is exported to a text file and then get shipped towards the cluster.

# ## Write the Training data to an External File

# In[ ]:


train_sentences_df = pd.read_excel("data/sentence-labels.xlsx")
train_sentences_df.head(2)


# In[ ]:


candidate_ids = train_sentences_df.candidate_id.astype(int).tolist()
candidate_ids


# In[ ]:


train_cands = (
    session
    .query(Candidate)
    .filter(Candidate.id.in_(candidate_ids))
    .all() 
)


# In[ ]:


lstm = reRNN(seed=100, n_threads=20)


# In[ ]:


rows = []
for c in tqdm.tqdm(train_cands):
    data, ends = lstm._preprocess_data([c], extend=True)
    rows.append({
        "data_str": ",".join([str(x) for x in data[0]]),
        "ends": ends[0],
        "candidate_id": c.id
    })
train_data = pd.DataFrame(rows)
train_data.head(2)


# In[ ]:


train_data = pd.merge(
    train_data,
    train_sentences_df[["candidate_id", "label"]],
    how='left'
)
train_data.head(2)


# In[ ]:


train_data.to_csv("data/lstm/train_data.tsv", sep="\t", index=False)


# ### Save the word dictionary to an External File

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with open("data/lstm/train_word_dict.csv", \'w\') as f:\n    output = csv.DictWriter(f, fieldnames=["Key", "Value"])\n    output.writeheader()\n    for key in tqdm.tqdm(lstm.word_dict.d):\n        output.writerow({\'Key\':key, \'Value\': lstm.word_dict.d[key]})')


# ## Write Dev data to an External File

# In[ ]:


dev_sentences_df = pd.read_excel("data/sentence-labels-dev.xlsx")
dev_sentences_df.head(2)


# In[ ]:


candidate_ids = dev_sentences_df.candidate_id.astype(int).tolist()
candidate_ids


# In[ ]:


dev_cands = (
    session
    .query(Candidate)
    .filter(Candidate.id.in_(candidate_ids))
    .all() 
)


# In[ ]:


rows = []
for c in tqdm.tqdm(dev_cands):
    data, ends = lstm._preprocess_data([c])
    rows.append({
        "data_str": ",".join([str(x) for x in data[0]]),
        "ends": ends[0],
        "candidate_id": c.id
    })
dev_data = pd.DataFrame(rows)
dev_data.head(2)


# In[ ]:


dev_data = pd.merge(
    dev_data,
    dev_sentences_df[["candidate_id", "label"]],
    how='left'
)
dev_data.head(2)


# In[ ]:


dev_data.to_csv("data/lstm/dev_data.tsv", sep="\t", index=False)


# # Train Sparse Logistic Regression Disc Model

# Here we train an SLR. To find the optimal hyperparameter settings this code uses a [random search](http://scikit-learn.org/stable/modules/grid_search.html) instead of iterating over all possible combinations of parameters. After the final model has been found, it is saved in the checkpoints folder to be loaded in the [next notebook](5.data-analysis.ipynb). Furthermore, the weights for the final model are output into a text file to be analyzed as well.

# In[ ]:


train_sentences_df = pd.read_excel("data/sentence-labels.xlsx")
train_sentences_df.head(2)


# In[ ]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(
    train_sentences_df.sentence.values
)
X


# In[ ]:


labels = [
    train_sentences_df.label.apply(lambda x: 1 if x > 0.5 else 0)
]

model_labels = [
    "all_LF_LR"
]
    
lr_grids = [
    {'C':np.linspace(1,10, num=4)} for _ in range(len(labels))
]
final_models = []


# In[ ]:


lr_model = LogisticRegression()


# In[ ]:


get_ipython().run_cell_magic('time', '', "for grid, y_labels in zip(lr_grids, labels):\n    fit_model = GridSearchCV(lr_model, \n                         grid, cv=10, n_jobs=3, \n                         verbose=1, scoring='roc_auc', return_train_score=True)\n    fit_model.fit(X, y_labels)\n    final_models.append(fit_model)")


# In[ ]:


print(len(final_models))


# In[ ]:


import matplotlib.pyplot as plt
for model, model_label in zip(final_models, model_labels):
    lr_result = pd.DataFrame(model.cv_results_)
    plt.plot(lr_result["param_C"], lr_result["mean_test_score"], label=model_label)
plt.legend()
plt.xlabel("C (regularization parameter)")
plt.ylabel("Mean Test Score")
plt.title("BOW Training CV (10-fold)")


# In[ ]:


for i, label in zip(range(len(final_models)), model_labels):
    lr_weights = pd.DataFrame(list(zip(final_models[i].best_estimator_.coef_[0], vectorizer.get_feature_names())), columns=["Weight", "Feature"])
    print(label)
    print(lr_weights.sort_values("Weight", ascending=False).head(10))
    print()


# In[ ]:


lr_marginals = []
for model in final_models:
    lr_marginals.append(model.best_estimator_.predict_proba(dev_X)[:,1])


# In[ ]:


print(pd.Series(final_models[0].best_estimator_.predict(dev_X)).value_counts())
print()
print(pd.Series(final_models[1].best_estimator_.predict(dev_X)).value_counts())
print()


# In[ ]:


for marginal, model_label in zip(lr_marginals,model_labels):
    filename = "vanilla_lstm/lstm_disease_gene_holdout/subsampled/lf_marginals/{}_dev_marginals.csv".format(model_label)
    pd.DataFrame(marginal,
             columns=["LR_Marginals"]
        ).to_csv(filename, index=False)

