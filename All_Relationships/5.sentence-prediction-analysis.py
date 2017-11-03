
# coding: utf-8

# # Lets See How The Disc Models Preformed

# This notebook is designed to analyze the disc models performance and to answer the question does Long Short Term Memory Neural Net (LSTM) outperform SparseLogisticRegression (SLR).

# ## MUST RUN AT THE START OF EVERYTHING

# Load the database and other helpful functions for analysis.

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import csv
import os

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, confusion_matrix
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
from snorkel.utils import get_as_dict
from snorkel.viewer import SentenceNgramViewer
from sqlalchemy import and_
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


# # Load the data

# Here is where we load the test dataset in conjunction with the previously trained disc models. Each algorithm will output a probability of a candidate being a true candidate.

# In[6]:


labeler = LabelAnnotator(lfs=[])


# In[7]:


get_ipython().run_cell_magic(u'time', u'', u'L_test = labeler.load_matrix(session,split=2)')


# In[8]:


L_test.shape


# In[9]:


model_marginals = pd.read_csv("stratified_data/lstm_disease_gene_holdout/lstm_one_percent.csv")
model_marginals["True_Labels"] = L_test.todense()
model_marginals["RNN_1_Predictions"] = model_marginals["RNN_marginals"].apply(lambda x: 1 if x > 0.5 else -1)
model_marginals = model_marginals.rename(index=str, columns={"RNN_marginals":"RNN_1_Marginals"})
model_marginals["RNN_10_Marginals"] = pd.read_csv("stratified_data/lstm_disease_gene_holdout/lstm_ten_percent.csv").values
model_marginals["RNN_10_Predictions"] = model_marginals["RNN_10_Marginals"].apply(lambda x: 1 if x > 0.5 else -1)


# In[10]:


test_set = pd.read_csv("stratified_data/lstm_disease_gene_holdout/test_candidates_sentences.csv")


# # Accuracy ROC

# From the probabilities calculated above, we can create a [Receiver Operator Curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) (ROC) graph to measure the false positive rate and the true positive rate at each calculated threshold.

# In[11]:


models = ["RNN_1_Marginals", "RNN_10_Marginals"]
model_colors = ["green", "magenta"]
model_labels = ["RNN_1%", "RNN_10%"]
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

for model_label, marginal_label, color in zip(model_labels, models, model_colors):
    fpr, tpr, _= roc_curve(model_marginals["True_Labels"], model_marginals[marginal_label])
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label="{} (area = {:0.2f})".format(model_label, model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Accuracy ROC')
plt.legend(loc="lower right")


# # Precision vs Recall Curve

# This code produces a [Precision-Recall](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) graph, which shows the trade off between [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) at each given probability threshold.

# In[12]:


models = ["RNN_1_Marginals", "RNN_10_Marginals"]
model_colors = ["green", "magenta"]
model_labels = ["RNN_1%", "RNN_10%"]


for model_label, marginal_label, color in zip(model_labels, models, model_colors):
    precision, recall, _ = precision_recall_curve(model_marginals["True_Labels"], model_marginals[marginal_label])
    model_precision = average_precision_score(model_marginals["True_Labels"], model_marginals[marginal_label])
    plt.plot(recall, precision, color=color, label="{} curve (area = {:0.2f})".format(model_label, model_precision))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision vs Recall')
plt.xlim([0, 1.01])
plt.ylim([0, 1.05])
plt.legend(loc="lower right")


# In[13]:


for model in ["RNN_1_Predictions", "RNN_10_Predictions"]:
    matrix = confusion_matrix(model_marginals["True_Labels"], model_marginals[model])
    chart1, ax1 = plt.subplots()
    plt.title(model)
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap="BuGn", xticklabels=["No","Yes"], yticklabels=["No", "Yes"], ax=ax1)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")


# ## LSTM Model Analysis

# In[14]:


condition = (model_marginals["RNN_10_Predictions"] == 1)&(model_marginals["True_Labels"] == -1)
indicies = set(model_marginals[condition].sort_values("RNN_10_Marginals", ascending=False).index)
model_marginals[condition].sort_values("RNN_10_Marginals", ascending=False).head(10)


# In[15]:


def insert(x, g_start, g_end, d_start, d_end, proba, d_cid, g_cid):
    if d_start == x[0] or g_start == x[0]:
        pos_str = "<span title=\"{}\" style=\"background-color: rgba(0,255,0,{})\">{}"
        neg_str = "<span title=\"{}\" style=\"background-color: rgba(255,0,0,{})\">{}"
        if proba > 0.5:
            return pos_str.format(d_cid, proba, x[1]) if d_start == x[0] else pos_str.format(g_cid, proba, x[1])
        else:
            return neg_str.format(d_cid, 1-proba, x[1]) if d_start == x[0] else neg_str.format(g_cid, 1-proba, x[1])
    elif d_end == x[0] or g_end == x[0]:
            return "{}</span>".format(x[1])
    else:
        return x[1]


# ## Look at the Sentences and the LSTM's predictions

# In[16]:


html_string = ""
counter = 0
condition = (model_marginals["RNN_10_Predictions"] == 1)&(model_marginals["True_Labels"] == -1)
for marginal in tqdm.tqdm(model_marginals[condition].itertuples()):
    cand = test_set.iloc[int(marginal[0])]
    
    counter += 1
    
    if counter == 200:
        break
        
    if counter > 0:
        gene_start = cand["gene_char_start"]
        gene_end = cand["gene_char_end"]
        disease_start = cand["disease_char_start"]
        disease_end = cand["disease_char_end"]
        proba = marginal[4]
        letters = []

        for x in enumerate(cand["sentence"]):
            letters.append(insert(x, gene_start, gene_end, disease_start, disease_end, proba, cand["disease_id"], cand["gene_id"]))

        html_string += "<div title=\"{}\">{}</div><br />".format(proba, ''.join(letters))


# In[17]:


with open("html/candidate_viewer.html", 'r') as f:
    display(HTML(f.read().format(html_string)))


# # Write Results to CSV

# In[ ]:


model_marginals.to_csv("stratified_data/lstm_disease_gene_holdout/test_marginals.csv", index=False)

