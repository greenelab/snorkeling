
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
import glob
import os

from IPython.core.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from scipy.stats import norm
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
from sklearn.metrics import auc, f1_score, confusion_matrix
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


import sys
sys.path.append('/home/danich1/Documents/snorkeling/snorkel/treedlib/treedlib')


# In[4]:


from snorkel.annotations import FeatureAnnotator, LabelAnnotator, load_marginals
from snorkel.models import Candidate, candidate_subclass
from snorkel.viewer import SentenceNgramViewer


# In[5]:


edge_type = "dg"


# In[6]:


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

# In[7]:


dev_sentence_df = pd.read_excel("data/sentence-labels-dev.xlsx")
dev_sentence_df = dev_sentence_df[dev_sentence_df.curated_dsh.notnull()]
dev_sentence_df = dev_sentence_df.sort_values("candidate_id")
dev_sentence_df.head(2)


# In[8]:


marginals_df = pd.read_table('data/disc_model_marginals.tsv')
marginals_df.head(2)


# In[9]:


dev_sentence_df.curated_dsh.value_counts()


# # Accuracy ROC

# From the probabilities calculated above, we can create a [Receiver Operator Curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html) (ROC) graph to measure the false positive rate and the true positive rate at each calculated threshold.

# In[10]:


model_aucs = {}


# In[11]:


plt.figure(figsize=(7,5))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

for model in marginals_df.columns:
    fpr, tpr, _= roc_curve(dev_sentence_df.curated_dsh.values, marginals_df[model])
    model_auc = auc(fpr, tpr)
    model_aucs[model] = model_auc
    plt.plot(fpr, tpr, label="{} (AUC = {:0.2f})".format(model, model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Accuracy ROC')
plt.legend(loc="lower right")


# ## Statistical Significance for ROC

# This block of code calculates the p-value for each ROC curve above. Each AUC is equivalent to the Mann-Whitney U statistic (aka Willcoxon Rank Sum Test). Calulating this statistic is easy and the equation can be found [here](https://www.quora.com/How-is-statistical-significance-determined-for-ROC-curves-and-AUC-values). Since more than 20 data points were used to generate the above ROC curves, normal approximation canm used to calculate the p-values. The equaltions for the normal approximation can be found [here](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test). Looking at the bottom dataframe, one can see there isn't strong evidence that the generated ROCs are greater than 0.50.

# In[12]:


class_dist_df = dev_sentence_df.curated_dsh.value_counts()
n1 = class_dist_df[0]
n2 = class_dist_df[1]
mu = (n1*n2)/2
sigma_u = np.sqrt((n1 * n2 * (n1+n2+1))/12)
print("mu: {:f}, sigma: {:f}".format(mu, sigma_u))


# In[13]:


model_auc_df = pd.DataFrame.from_dict(model_aucs, orient='index')
model_auc_df = model_auc_df.rename(index=str, columns={0:'auroc'})
model_auc_df.head(2)


# In[14]:


model_auc_df['u'] = model_auc_df.auroc.apply(lambda x: x*n1*n2)
model_auc_df['z_u'] = model_auc_df.u.apply(lambda z_u: (z_u- mu)/sigma_u)
model_auc_df.head(2)


# In[15]:


model_auc_df['p_value'] = model_auc_df.z_u.apply(lambda z_u: norm.sf(z_u, loc=0, scale=1))
model_auc_df


# # Precision vs Recall Curve

# This code produces a [Precision-Recall](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html) graph, which shows the trade off between [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) at each given probability threshold.

# In[16]:


plt.figure(figsize=(7,5))

positive_class = dev_sentence_df.curated_dsh.values.sum()/dev_sentence_df.shape[0]
plt.plot([0,1], [positive_class, positive_class], color='grey', 
         linestyle='--', label='Baseline (AUC = {:0.2f})'.format(positive_class))

for model in marginals_df.columns:
    precision, recall, _ = precision_recall_curve(dev_sentence_df.curated_dsh.values, marginals_df[model])
    model_auc = auc(recall, precision)
    plt.plot(recall, precision, label="{} curve (AUC = {:0.2f})".format(model, model_auc))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision vs Recall')
plt.xlim([0, 1.01])
plt.ylim([0, 1.05])
plt.legend(loc="lower right")


# # LSTM BENCHMARKING

# In[ ]:


import glob
import os
f, ax = plt.subplots(3,3, sharex='col', sharey='row', figsize=(12,8))
color_map = {
    "0.25":"blue",
    "0.5":"green",
    "0.75":"brown"
}
#Make graph learning rate by dimension
index = 1
for row, dimension in enumerate([100, 250,500]):
    for col, lr in enumerate([0.0005, 0.001, 0.002]):
        for diagnostics_file in glob.glob(r"vanilla_lstm/benchmark/no_dropout/lstm_{}_*_{}.tsv".format(lr, dimension)):
            benchmark_label = os.path.splitext(os.path.basename(diagnostics_file))[0]
            benchmark_label = re.search(r'_(\d\.\d\d?)_', benchmark_label).group(1)
            
            diagnostics_df = pd.read_table(diagnostics_file)
            ax[col][row].plot(diagnostics_df["train_loss"], label="_no_legend_")
            ax[col][row].plot(diagnostics_df["val_loss"], label="Keep: {}".format(benchmark_label), color=color_map[benchmark_label])
        ax[col][row].set_title("Learn Rate: {}, Dim: {}".format(lr, dimension))
        if col==0 and row == 0:
            f.legend()
        index += 1
f.text(0.5, 0.04, 'Epochs', ha='center', va='center')
f.text(0.06, 0.5, 'Log Loss', ha='center', va='center', rotation='vertical')
f.suptitle("LSTM Benchmark", fontsize=14)
#plt.xlim([0,105])
#plt.ylim([0,0.3])
#plt.xlabel("Epochs")
#plt.ylabel("Log Loss")
plt.show()


# In[ ]:


import glob
import os
f, ax = plt.subplots(2,2, sharex='col', sharey='row', figsize=(12,8))
color_map = {
    "1":"red",
    "0.25":"blue",
    "0.5":"green",
    "0.75":"brown"
}
legend_dict = {}
#Make graph learning rate by dimension
index = 1
for row, dimension in enumerate([250,500]):
    for col, lr in enumerate([0.001, 0.002]):
        for diagnostics_file in glob.glob(r"vanilla_lstm/benchmark/dropout/lstm_{}_*_{}.tsv".format(lr, dimension)):
            benchmark_label = os.path.splitext(os.path.basename(diagnostics_file))[0]
            benchmark_label = re.search(r'_(\d\.?\d?\d?)_', benchmark_label).group(1)

            diagnostics_df = pd.read_table(diagnostics_file)
            ax[col][row].plot(diagnostics_df["train_loss"], color=color_map[benchmark_label], label='_nolegend_')
            ax[col][row].plot(diagnostics_df["val_loss"], label="Keep: {}".format(benchmark_label), color=color_map[benchmark_label])
        ax[col][row].set_title("Learn Rate: {}, Dim: {}".format(lr, dimension))
        #ax[col][row].legend()
        index += 1
        if col == 0 and row == 0:
            f.legend()
        #ax[col][row].set_ylim([0.175,0.2])
f.text(0.5, 0.04, 'Epochs', ha='center', va='center')
f.text(0.06, 0.5, 'Log Loss', ha='center', va='center', rotation='vertical')
f.suptitle("LSTM Benchmark", fontsize=14)
plt.show()


# ## LSTM Model Analysis

# In[ ]:


marginal_criteria = "RNN_100_Marginals"
model_predictions = model_marginals[marginal_criteria].apply(lambda x: 1 if x > 0.5 else -1)
model_predictions.head(10)


# In[ ]:


condition = (model_predictions == 1)&(model_marginals["True_Labels"] == -1)
model_marginals[condition].sort_values(marginal_criteria, ascending=False).head(10)


# In[ ]:


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

# In[ ]:


html_string = ""
counter = 0
sorted_marginals = (
     model_marginals.query("@model_predictions ==1 & True_Labels == -1")
                    .sort_values(marginal_criteria, ascending=False)
)

for cand_index, marginal in tqdm.tqdm(sorted_marginals[marginal_criteria].iteritems()):
    cand = test_set.iloc[cand_index]
    counter += 1
    
    if counter == 10:
        break
        
    if counter > 0:
        gene_start = cand["gene_char_start"]
        gene_end = cand["gene_char_end"]
        disease_start = cand["disease_char_start"]
        disease_end = cand["disease_char_end"]
        letters = []

        for x in enumerate(cand["sentence"]):
            letters.append(insert(x, gene_start, gene_end, disease_start, disease_end, marginal, cand["disease_id"], cand["gene_id"]))

        html_string += "<div title=\"{}\">{}</div><br />".format(marginal, ''.join(letters))


# In[ ]:


with open("html/candidate_viewer.html", 'r') as f:
    display(HTML(f.read().format(html_string)))


# # Write Results to CSV

# In[ ]:


model_marginals.to_csv("stratified_data/lstm_disease_gene_holdout/total_test_marginals.csv", index=False)

