
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

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score
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


featurizer = FeatureAnnotator()
labeler = LabelAnnotator(lfs=[])


# In[7]:


get_ipython().run_cell_magic(u'time', u'', u'L_test = labeler.load_matrix(session,split=2)\nF_test = featurizer.load_matrix(session, split=2)')


# In[8]:


model_marginals = pd.read_csv("disc_marginals.csv")
lr_df = pd.read_csv("LR_model.csv")


# # Accuracy ROC

# From the probabilities calculated above, we can create a [Receiver Operator Curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) (ROC) graph to measure the false positive rate and the true positive rate at each calculated threshold.

# In[9]:


models = ["LR_Marginals", "RNN_1_Marginals", "RNN_10_Marginals", "RNN_Full_Marginals"]
model_colors = ["darkorange", "red", "green", "magenta"]
model_labels = ["LogReg", "RNN_1%", "RNN_10%", "RNN_100%"]
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

for model_label, marginal_label, color in zip(model_labels, models, model_colors):
    fpr, tpr, _= roc_curve(model_marginals["True Labels"], model_marginals[marginal_label])
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label="{} (area = {:0.2f})".format(model_label, model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Accuracy ROC')
plt.legend(loc="lower right")


# # Precision vs Recall Curve

# This code produces a [Precision-Recall](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) graph, which shows the trade off between [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) at each given probability threshold.

# In[10]:


models = ["LR_Marginals", "RNN_1_Marginals", "RNN_10_Marginals", "RNN_Full_Marginals"]
model_colors = ["darkorange", "red", "green", "magenta"]
model_labels = ["LogReg", "RNN_1%", "RNN_10%", "RNN_100%"]

for model_label, marginal_label, color in zip(model_labels, models, model_colors):
    precision, recall, _ = precision_recall_curve(model_marginals["True Labels"], model_marginals[marginal_label])
    model_precision = average_precision_score(model_marginals["True Labels"], model_marginals[marginal_label])
    plt.plot(recall, precision, color=color, label="{} curve (area = {:0.2f})".format(model_label, model_precision))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision vs Recall')
plt.xlim([0, 1.01])
plt.ylim([0, 1.05])
plt.legend(loc="lower right")


# # Confusion Matrix

# This code below produces a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix) for futher ML analysis/

# In[58]:


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(model_marginals["LR_Predictions"], model_marginals["True Labels"]).ravel()
ax = sns.heatmap([[tp, fn],[fp,tn]], annot=True, fmt='d', cmap="GnBu")
ax.set_xticklabels(["True", "False"])
ax.set_yticklabels(["False", "True"])
plt.xlabel("Classifier Prediction")
plt.ylabel("True Labels")
plt.title("LR Confusion Matrix")


# In[59]:


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(model_marginals["RNN_1_Predictions"], model_marginals["True Labels"]).ravel()
ax = sns.heatmap([[tp, fn],[fp,tn]], annot=True, fmt='d', cmap="GnBu")
ax.set_xticklabels(["True", "False"])
ax.set_yticklabels(["False", "True"])
plt.xlabel("Classifier Prediction")
plt.ylabel("True Labels")
plt.title("RNN 1% Confusion Matrix")


# In[60]:


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp= confusion_matrix(model_marginals["RNN_10_Predictions"], model_marginals["True Labels"]).ravel()
ax = sns.heatmap([[tp, fn],[fp,tn]], annot=True, fmt='d', cmap="GnBu")
ax.set_xticklabels(["True", "False"])
ax.set_yticklabels(["False", "True"])
plt.xlabel("Classifier Prediction")
plt.ylabel("True Labels")
plt.title("RNN 10% Confusion Matrix")


# In[61]:


from sklearn.metrics import confusion_matrix
tn, fp, fn, tp= confusion_matrix(model_marginals["RNN_Full_Predictions"], model_marginals["True Labels"]).ravel()
ax = sns.heatmap([[tp, fn],[fp,tn]], annot=True, fmt='d', cmap="GnBu")
ax.set_xticklabels(["True", "False"])
ax.set_yticklabels(["False", "True"])
plt.xlabel("Classifier Prediction")
plt.ylabel("True Labels")
plt.title("RNN 100% Confusion Matrix")


# # Error Analysis

# This code shows the amount of true positives, false positives, true negatives and false negatives.

# In[62]:


result_category = "fp"
if result_category == "tp":
    lr_cond = (model_marginals["LR_Predictions"] == 1)&(model_marginals["True Labels"] == 1)
    rnn1_cond = (model_marginals["RNN_1_Predictions"] == 1)&(model_marginals["True Labels"] == 1)
    rnn10_cond = (model_marginals["RNN_10_Predictions"] == 1)&(model_marginals["True Labels"] == 1)
    rnn100_cond = (model_marginals["RNN_Full_Predictions"] == 1)&(model_marginals["True Labels"] == 1)
elif result_category == "fp":
    lr_cond = (model_marginals["LR_Predictions"] == 1)&(model_marginals["True Labels"] == -1)
    rnn1_cond = (model_marginals["RNN_1_Predictions"] == 1)&(model_marginals["True Labels"] == -1)
    rnn10_cond = (model_marginals["RNN_10_Predictions"] == 1)&(model_marginals["True Labels"] == -1)
    rnn100_cond = (model_marginals["RNN_Full_Predictions"] == 1)&(model_marginals["True Labels"] == -1)
elif result_category == "tn":
    lr_cond = (model_marginals["LR_Predictions"] == -1)&(model_marginals["True Labels"] == -1)
    rnn1_cond = (model_marginals["RNN_1_Predictions"] == -1)&(model_marginals["True Labels"] == -1)
    rnn10_cond = (model_marginals["RNN_10_Predictions"] == -1)&(model_marginals["True Labels"] == -1)
    rnn100_cond = (model_marginals["RNN_Full_Predictions"] == -1)&(model_marginals["True Labels"] == -1)
elif result_category == "fn":
    lr_cond = (model_marginals["LR_Predictions"] == -1)&(model_marginals["True Labels"] == 1)
    rnn1_cond = (model_marginals["RNN_1_Predictions"] == -1)&(model_marginals["True Labels"] == 1)
    rnn10_cond = (model_marginals["RNN_10_Predictions"] == -1)&(model_marginals["True Labels"] == 1)
    rnn100_cond = (model_marginals["RNN_Full_Predictions"] == -1)&(model_marginals["True Labels"] == 1)
else:
    print ("Please re-run cell with correct options")


# In[14]:


display_columns = ["LR_Marginals", "RNN_1_Marginals", "RNN_10_Marginals", "RNN_Full_Marginals", "True Labels"]


# ## LR

# In[28]:


model_marginals[lr_cond].sort_values("LR_Marginals", ascending=False).head(10)[display_columns]


# In[29]:


cand_index = list(model_marginals[lr_cond].sort_values("LR_Marginals", ascending=False).head(10).index)
lr_cands = [L_test.get_candidate(session, i) for i in cand_index]


# In[30]:


print "Category: {}".format(result_category)
print 
for cand, cand_ind in zip(lr_cands, cand_index):
    text = cand[0].get_parent().text
    text = re.sub(cand[0].get_span().replace(")", "\)"), "--[[{}]]D--".format(cand[0].get_span()), text)
    text = re.sub(cand[1].get_span().replace(")", "\)"), "--[[{}]]G--".format(cand[1].get_span()), text)
    print cand_ind
    print "Candidate: ", cand
    print
    print "Text: \"{}\"".format(text)
    print
    print "--------------------------------------------------------------------------------------------"
    print


# In[ ]:


F_cand_index = 137865
print "Confidence Level: ", model_marginals["LR_Marginals"][F_cand_index]


# In[ ]:


F_cand_index = 137865
lr_df.iloc[F_test[F_cand_index, :].nonzero()[1]].sort_values("Weight", ascending=False)


# In[ ]:


cand = session.query(Candidate).filter(Candidate.id == L_test.get_candidate(session, 137865).id).one()
print cand
xmltree = corenlp_to_xmltree(get_as_dict(cand.get_parent()))
xmltree.render_tree(highlight=[range(cand[0].get_word_start(), cand[0].get_word_end() + 1), range(cand[1].get_word_start(), cand[1].get_word_end()+1)])


# ## LSTM 1% Sub-Sampling

# In[31]:


model_marginals[rnn1_cond].sort_values("RNN_1_Marginals", ascending=False).head(10)[display_columns]


# In[32]:


cand_index = list(model_marginals[rnn1_cond].sort_values("RNN_1_Marginals", ascending=False).head(10).index)
lr_cands = [L_test.get_candidate(session, i) for i in cand_index]


# In[33]:


print "Category: {}".format(result_category)
print 
for cand in lr_cands:
    text = cand[0].get_parent().text
    text = re.sub(cand[0].get_span().replace(")", "\)"), "--[[{}]]D--".format(cand[0].get_span()), text)
    text = re.sub(cand[1].get_span().replace(")", "\)"), "--[[{}]]G--".format(cand[1].get_span()), text)
    print "Candidate: ", cand
    print
    print "Text: \"{}\"".format(text)
    print
    print "--------------------------------------------------------------------------------------------"
    print


# ## LSTM 10% Sub-Sampling

# In[34]:


model_marginals[rnn10_cond].sort_values("RNN_10_Marginals", ascending=False).head(10)[display_columns]


# In[35]:


cand_index = list(model_marginals[rnn10_cond].sort_values("RNN_10_Marginals", ascending=False).head(10).index)
lr_cands = [L_test.get_candidate(session, i) for i in cand_index]


# In[36]:


print "Category: {}".format(result_category)
print 
for cand in lr_cands:
    text = cand[0].get_parent().text
    text = re.sub(cand[0].get_span().replace(")", "\)"), "--[[{}]]D--".format(cand[0].get_span()), text)
    text = re.sub(cand[1].get_span().replace(")", "\)"), "--[[{}]]G--".format(cand[1].get_span()), text)
    print "Candidate: ", cand
    print
    print "Text: \"{}\"".format(text)
    print
    print "--------------------------------------------------------------------------------------------"
    print


# # FULL LSTM

# In[63]:


model_marginals[rnn100_cond].sort_values("RNN_Full_Marginals", ascending=False).head(10)[display_columns]


# In[64]:


cand_index = list(model_marginals[rnn100_cond].sort_values("RNN_Full_Marginals", ascending=False).head(10).index)
lr_cands = [L_test.get_candidate(session, i) for i in cand_index]


# In[66]:


print "Category: {}".format(result_category)
print 
for cand in lr_cands:
    text = cand[0].get_parent().text
    text = re.sub(cand[0].get_span().replace(")", "\)"), "--[[{}]]D--".format(cand[0].get_span()), text)
    text = re.sub(cand[1].get_span().replace(")", "\)"), "--[[{}]]G--".format(cand[1].get_span()), text)
    print "Candidate: ", cand
    print
    print "Text: \"{}\"".format(text)
    print
    print "--------------------------------------------------------------------------------------------"
    print


# # Write Results to TSV

# In[87]:


field_names = ["Disease ID", "Disease Char Start", "Disease Char End", "Gene ID", "Gene Char Start", "Gene Char End", "Sentence", "Prediction"]
with open("test_set_results.tsv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    for i in tqdm.tqdm(model_marginals.index):
        cand = L_test.get_candidate(session, i)
        row = {
                "Disease ID": cand.Disease_cid, "Disease Char Start":cand[0].char_start, 
                "Disease Char End": cand[0].char_end, "Gene ID": cand.Gene_cid, 
                "Gene Char Start":cand[1].char_start, "Gene Char End":cand[1].char_end, 
                "Sentence": cand.get_parent().text, "Prediction": model_marginals.iloc[i]["RNN_Full_Marginals"]}
        writer.writerow(row)

