
# coding: utf-8

# # Lets See How The Disc Models Preformed

# This notebook is designed to analyze the disc models performance and to answer the question does Long Short Term Memory Neural Net (LSTM) outperform SparseLogisticRegression (SLR).

# ## MUST RUN AT THE START OF EVERYTHING

# Load the database and other helpful functions for analysis.

# In[ ]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score
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


# # Load the data

# Here is where we load the test dataset in conjunction with the previously trained disc models. Each algorithm will output a probability of a candidate being a true candidate.

# In[ ]:


featurizer = FeatureAnnotator()
labeler = LabelAnnotator(lfs=[])


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'L_test = labeler.load_matrix(session,split=2)\nF_test = featurizer.load_matrix(session, split=2)')


# In[ ]:


lr_model = SparseLogisticRegression()
lstm_model = reRNN(seed=100, n_threads=4)


# In[ ]:


lr_model.load(save_dir='checkpoints/grid_search/', model_name="SparseLogisticRegression_1")
lstm_model.load(save_dir='checkpoints/rnn', model_name="RNN")


# # Export the Data for analysis below

# Export the necessary data such as top predicted candidates and the traning marginals from the SparseLogisticRegression (SLR) model. LSTM and deep learning data to come.

# In[ ]:


lr_marginals = lr_model.marginals(F_test)
rnn_marginals = lstm.marginals(F_test)
marginal_df = pd.DataFrame([lr_marginals, rnn_marginals], columns=["LR_Marginals", "RNN_marginals"])
marginal_df.to_csv("disc_marginals.csv", index=False)


# In[ ]:


model_marginals = pd.read_csv("disc_marginals.csv")
top_pos_predict_model_marginals = model_marginals.sort_values("LR_Marginals", ascending=False).head(10)
top_neg_predict_model_marginals = model_marginals.sort_values("LR_Marginals", ascending=True).head(10)


# In[ ]:


from collections import defaultdict
pos_feature_freq = defaultdict(int)
for index in tqdm.tqdm(top_pos_predict_model_marginals.index):
    top_match_feat = F_test[index,:].nonzero()[1]
    for feature in lr_df["Feature"][top_match_feat]:
        pos_feature_freq[feature] += 1
pos_features_df = pd.DataFrame(pos_feature_freq.items(), columns=["Feature", "Frequency"])


# In[ ]:


from collections import defaultdict
neg_feature_freq = defaultdict(int)
for index in tqdm.tqdm(top_neg_predict_model_marginals.index):
    top_match_feat = F_test[index,:].nonzero()[1]
    for feature in lr_df["Feature"][top_match_feat]:
        neg_feature_freq[feature] += 1
neg_features_df = pd.DataFrame(neg_feature_freq.items(), columns=["Feature", "Frequency"])


# In[ ]:


pos_features_df.sort_values("Frequency", ascending=False).to_csv('POS_LR_Feat.csv', index=False)


# In[ ]:


neg_features_df.sort_values("Frequency", ascending=False).to_csv("NEG_LR_Feat.csv", index=False)


# # Error Analysis

# This code shows the amount of true positives, false positives, true negatives and false negatives.

# In[ ]:


_, _, _, _ = lr_model.error_analysis(session, F_test, L_test)


# In[ ]:


_, _, _, _ = lstm_model.error_analysis(session, F_test, L_test)


# # Accuracy ROC

# From the probabilities calculated above, we can create a [Receiver Operator Curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) (ROC) graph to measure the false positive rate and the true positive rate at each calculated threshold.

# In[ ]:


plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

for model_marginals, color in zip(["LR_Marginals", "RNN_marginals"], ["darkorange", "red"]):
    fpr, tpr, _= roc_curve(L_test[0:].todense(), marginal_df[model_marginals])
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label="{} curve (area = {0.2f})".format(model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Accuracy ROC')
plt.legend(loc="lower right")


# # Precision vs Recall Curve

# This code produces a [Precision-Recall](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) graph, which shows the trade off between [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) at each given probability threshold.

# In[ ]:


for model_marginals, color in zip(["LR_Marginals", "RNN_marginals"], ["darkorange", "red"]):
    precision, recall, _=  precision_recall_curve(L_test[0:].todense(), marginal_df[model_marginals])
    model_f1 = f1_score(L_test[0:].todense(), marginal_df[model_marginals])
    plt.plot(fpr, tpr, color=color, label="{} curve (area = {0.2f})".format(model_f1))

plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision vs Recall')
plt.legend(loc="lower right")


# # LR Model Details

# ## Global Picture of the Model

# Taking a deeper look into the SLR model, we can see that the highest weighted features are all cancer related. This leads one to think that majority of these abstracts are cancer related. Looking at the distribution of the weights it follows a normal distribution with a significant number of zeros. These zeros are desired which means majority of these feautres do not have significant weight to determine if a candidate is true or not. Furthermore, using the last cell of this block we can take a close look at the [dependency tree](https://nlp.stanford.edu/software/stanford-dependencies.shtml) stanford's core nlp used for generating features.

# In[ ]:


lr_df = pd.read_csv("LR_model.csv")


# In[ ]:


weight_df = lr_df.sort_values("Weight", ascending=False, kind='mergesort')
weight_df.head(15)


# In[ ]:


n, bins, patches = plt.hist(weight_df["Weight"])
plt.xlabel('Weight')
plt.ylabel('Count')
plt.title('Distribution of LR Weights')


# In[ ]:


cand = session.query(Candidate).filter(Candidate.id==674118).all()
print cand
print cand[0].get_parent()


# In[ ]:


cand = session.query(Candidate).filter(Candidate.id == 19841894).one()
print cand
xmltree = corenlp_to_xmltree(get_as_dict(cand.get_parent()))
xmltree.render_tree(highlight=[range(cand[0].get_word_start(), cand[0].get_word_end() + 1), range(cand[1].get_word_start(), cand[1].get_word_end()+1)])


# ## Taking a Deeper Look into the Model

# From the above cells we saw that the SLR model is somewhat behaving as we expected. This section here attempts to dive deeper into the model and examine the top predicted candidates (both positive and negative). After gathering each candidate, we take a consensus of all the features these top candidates share, so we can have a better understanding on how these predictions came to be.

# In[ ]:


pos_features_df = pd.read_csv("POS_LR_Feat.csv")
neg_features_df = pd.read_csv("NEG_LR_Feat.csv")


# In[ ]:


pos_features_df.head(10)


# In[ ]:


neg_features_df.head(10)


# # RNN Model Details

# TBD
