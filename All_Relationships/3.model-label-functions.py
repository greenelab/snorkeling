
# coding: utf-8

# # Train the Generative Model for Candidate Labeling

# This notebook is designed to run a generative model that snorkel uses to probabilistically label each candidate. (1 for positive label and -1 for negative label). Using this generative model, we will test the hypothesis: **modeling correlation structure between label functions provides better precision and recall than the conditionally independent model.**

# ## MUST RUN AT THE START OF EVERYTHING

# Import the necessary modules and set up the database for database operations.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter, OrderedDict, defaultdict
import os
from tqdm import tqdm_notebook

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score


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
from snorkel.annotations import load_gold_labels
from snorkel.annotations import FeatureAnnotator, LabelAnnotator, save_marginals
from snorkel.learning import GenerativeModel
from snorkel.learning.structure import DependencySelector
from snorkel.learning.utils import MentionScorer
from snorkel.models import Candidate, FeatureKey, candidate_subclass, Label
from snorkel.utils import get_as_dict
from utils.compound_gene_lf import CG_LFS
from utils.disease_gene_lf import DG_LFS


# In[4]:


edge_type = "cg"


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

# This code will load the corresponding label matricies that were generated in the previous notebook ([Notebook 2](2.data-labeler.ipynb)). This notebook has three matricies which are broken down as follows:
# 
# |Dataset|Size|Description|
# |:-----|-----|:-----|
# |L_train|50,000|Randomly sampled from our 2,700,000 training set|
# |L_dev|10,000|Randomly sampled from our 700,000 dev set. Only 200 have been hand labeled|
# |L_train_labeled|919|Have been hand labled from training set and is separate from (L_train).|

# In[6]:


train_candidate_ids = pd.read_excel('data/compound_gene/sentence_labels.xlsx').candidate_id.astype(int).tolist()
train_candidate_ids[0:10]


# In[7]:


dev_data_df = pd.read_excel("data/compound_gene/sentence_labels_dev.xlsx")
dev_data_df = dev_data_df[dev_data_df.curated_dsh.notnull()]
dev_data_df = dev_data_df.sort_values("candidate_id")
dev_candidate_ids = list(map(int, dev_data_df.candidate_id.values))
#dev_candidate_ids = np.loadtxt('data/compound_gene/labeled_dev_candidates.txt').astype(int).tolist()
print("Total Hand Labeled Dev Sentences: {}".format(len(dev_candidate_ids)))


# In[8]:


get_ipython().run_cell_magic('time', '', 'labeler = LabelAnnotator(lfs=[])\n\n# Only grab candidates that have labels\ncids = session.query(Candidate.id).filter(Candidate.id.in_(train_candidate_ids))\nL_train = labeler.load_matrix(session, cids_query=cids)\n\ncids = session.query(Candidate.id).filter(Candidate.id.in_(dev_candidate_ids))\nL_dev = labeler.load_matrix(session,cids_query=cids)')


# In[9]:


train_hand_df = pd.read_excel('data/compound_gene/sentence_labels_train_hand.xlsx')
train_hand_df = train_hand_df[train_hand_df.curated_dsh.notnull()]
train_cids = train_hand_df.candidate_id.astype(int).tolist()

cids = session.query(Candidate.id).filter(Candidate.id.in_(train_cids))
L_train_labeled = labeler.load_matrix(session, cids_query=cids)


# In[10]:


print("Total Number of Hand Labeled Candidates: {}\n".format(len(train_cids)))
print("Distribution of Labels:")
print(train_hand_df.curated_dsh.value_counts())


# In[11]:


print("Total Size of Train Data: {}".format(L_train.shape[0]))
print("Total Number of Label Functions: {}".format(L_train.shape[1]))


# # Train the Generative Model

# Here is the first step in classifying candidate sentences. We train a generative model to probabilistically label each training sentence. This means the model assigns a probability to each sentence indicating whether or not it mentions a given relatinoship (> 0.5 if yes, 0.5 < if no). The generative model snorkel uses is a [factor graph](http://deepdive.stanford.edu/assets/factor_graph.pdf) and further information on this model can be found in their paper [here](https://arxiv.org/abs/1711.10160).
# 
# The following code below trains two different generative models. One model follows the assumption that each label function is independent of each other, while the other model assumes there are dependancies between each function (e.g. $L_{1}$ correlates with $L_{2}$).

# In[12]:


def get_columns(lf_hash, lf_name):
    return list(
    map(lambda x: L_train.key_index[x[0]],
    session.query(L_train.annotation_key_cls.id)
         .filter(L_train.annotation_key_cls.name.in_(list(lf_hash[lf_name].keys())))
         .all()
        )
    )


# In[13]:


cg_db = get_columns(CG_LFS, "CbG_DB")
cg_text = get_columns(CG_LFS, "CbG_TEXT")
dg_text = get_columns(DG_LFS, "DaG_TEXT")

lfs_columns = [cg_db, cg_text,dg_text, cg_db+cg_text, cg_db+cg_text+dg_text]
model_names = ["CbG_DB", "CbG_TEXT", "DaG_TEXT", "CbG_ALL", "DaG_CbG_ALL"]


# In[14]:


indep_models = []
for columns in lfs_columns:
    #Conditionally independent Generative Model
    indep_gen_model = GenerativeModel()
    indep_gen_model.train(
        L_train[:, columns],
        epochs=10,
        decay=0.95,
        step_size=0.1 / L_train[:, columns].shape[0],
        reg_param=1e-6,
        threads=50,
    )
    indep_models.append(indep_gen_model)


# In[15]:


dep_models = []
for columns in lfs_columns:
    # select the dependancies from the label matrix
    ds = DependencySelector()
    deps = ds.select(L_train[:, columns], threshold=0.1)
    print(len(deps))

    # Model each label function and the underlying correlation structure
    gen_model = GenerativeModel(lf_propensity=True)
    gen_model.train(
        L_train[:, columns],
        epochs=10,
        decay=0.95,
        step_size=0.1 / L_train[:, columns].shape[0],
        reg_param=1e-6,
        threads=50,
        deps=deps
    )
    
    dep_models.append(gen_model)


# # Generative Model Statistics

# Now that both models have been trained, the next step is to generate some statistics about each model. The two histograms below show a difference between both models' output. The conditionally independent model (CI) predicts more negative candidates compared to the dependancy aware model (DA).

# In[16]:


get_ipython().run_cell_magic('time', '', 'train_marginals_indep_df = pd.DataFrame(\n    np.array([model.marginals(L_train[:, columns]) for model, columns in zip(indep_models, lfs_columns)]).T,\n    columns=model_names\n)\n\ntrain_marginals_dep_df = pd.DataFrame(\n    np.array([model.marginals(L_train[:, columns]) for model, columns in zip(dep_models, lfs_columns)]).T,\n    columns=model_names\n)')


# In[17]:


f, axs = plt.subplots(len(lfs_columns), 1, figsize=(10, 5), sharex=True)

for columns, ax in zip(model_names, axs):
    sns.distplot(train_marginals_indep_df[columns], ax=ax, kde=False, axlabel=False)
    ax.set_ylabel(columns, fontsize=12)

f.suptitle("CI Training Set Marginals")
plt.xlabel("Probability of Positive Class", fontsize=12)
plt.show()


# In[18]:


f, axs = plt.subplots(len(lfs_columns), 1, figsize=(10, 5), sharex=True)

for columns, ax in zip(model_names, axs):
    sns.distplot(train_marginals_dep_df[columns], ax=ax, kde=False, axlabel=False)
    ax.set_ylabel(columns, fontsize=12)

f.suptitle("DA Training Set Marginals")
plt.xlabel("Probability of Positive Class", fontsize=12)
plt.show()


# # Training Set Statistics

# Taking a closer look into the training set predictions, we can see how each label function individually performed. The two dataframes below contain the follwoing information: number of candidate sentences a label function has labeled (coverage), number of candidate sentences a label function agreed with another label function (overlaps), number of candidates a label function disagreed with another label function (conflicts), and lastly, the accuracy each label function has after training the generative model (Learned Acc).

# In[19]:


# Generate Statistics of Generative Model
indep_learned_stats_df = indep_models[-1].learned_lf_stats()
learned_stats_df = dep_models[-1].learned_lf_stats()


# In[20]:


indep_results_df = L_train[:, lfs_columns[-1]].lf_stats(session, est_accs=indep_learned_stats_df['Accuracy'])
indep_results_df


# In[21]:


results_df = L_train[:, lfs_columns[-1]].lf_stats(session, est_accs=learned_stats_df['Accuracy'])
results_df


# The following bar charts below depict the weights the generative model assigns to each label function. The conditional independent model relies heavily on LF_HETNET_ABSENT and LF_NO_CONCLUSION, while the dependancy aware model relies more on the database-backed label functions. Ultimately, the DA model emphasizes more postive labels compared to the CI model. 

# In[22]:


test_df = pd.concat([
    results_df[["Learned Acc."]].assign(model="DA"),
    indep_results_df[["Learned Acc."]].assign(model="CI"), 
])
test_df = test_df.reset_index()
test_df.head(2)


# In[51]:


fig, ax = plt.subplots(figsize=(10,11))
sns.barplot(ax=ax,y="index", x="Learned Acc.", hue="model", data=test_df, palette=sns.color_palette("muted"))


# ## F1 Score of Dev Set

# Moving from the training set, we now can look at how well these models can predict our small dev set. Looking at the chart below, the conditionally independent model doesn't perform well compared to the dependency aware model. In terms of f1 score there is about a .2 difference, which provides evidence towards the dependency model performing better.

# In[24]:


dev_data_labels = dev_data_df.curated_dsh.replace({0:-1})


# In[25]:


indep_results = {}
for columns, models, name in zip(lfs_columns, indep_models, model_names):
    print(name)
    indep_results[name] = models.error_analysis(session, L_dev[:, columns], dev_data_labels)


# In[26]:


dep_results = {}
for columns, models, name in zip(lfs_columns, dep_models, model_names):
    print(name)
    dep_results[name] = models.error_analysis(session, L_dev[:, columns], dev_data_labels)


# In[27]:


dev_marginals_indep_df = pd.DataFrame(
    np.array([model.marginals(L_dev[:, columns]) for model, columns in zip(indep_models, lfs_columns)]).T,
    columns=model_names
)

dev_marginals_dep_df = pd.DataFrame(
    np.array([model.marginals(L_dev[:, columns]) for model, columns in zip(dep_models, lfs_columns)]).T,
    columns=model_names
)


# In[28]:


plt.figure(figsize=(10,6))
plt.plot([0,1], [0,1], linestyle='--', color='grey', label="Random (AUC = 0.50)")

for marginal, model_label in zip(dev_marginals_indep_df.columns, model_names):
    fpr, tpr, threshold = roc_curve(dev_data_labels, dev_marginals_indep_df[marginal])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model_label+" (AUC = {:0.2f})".format(area))
    
plt.title("ROC of INDEP Generative Models")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()


# In[45]:


plt.figure(figsize=(10,6))
plt.plot([0,1], [0,1], linestyle='--', color='grey', label="Random (AUC = 0.50)")

for marginal, model_label in zip(dev_marginals_dep_df.columns, model_names):
    fpr, tpr, threshold = roc_curve(dev_data_labels, dev_marginals_dep_df[marginal])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model_label+" (AUC = {:0.2f})".format(area))
    
plt.title("ROC of DEP Generative Models")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()


# In[30]:


plt.figure(figsize=(10,6))
positive_class = dev_data_df.curated_dsh.values.sum()/dev_data_df.shape[0]
plt.plot([0,1], [positive_class, positive_class], color='grey', 
         linestyle='--', label='Baseline (AUC = {:0.2f})'.format(positive_class))

for marginal, model_label in zip(dev_marginals_indep_df.columns, model_names):
    precision, recall, threshold = precision_recall_curve(dev_data_labels, dev_marginals_indep_df[marginal])
    area = auc(recall, precision)
    plt.plot(recall, precision, label=model_label+" (AUC = {:0.2f})".format(area))
    
plt.title("Precision Recall Curve of INDEP Generative Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()


# In[31]:


L_dev[:, lfs_columns[-1]].lf_stats(session, dev_data_labels, test_df.query("model=='CI'")["Learned Acc."])


# ## Comparison Between The Amount of Label Functions

# In[32]:


#gen_model_history_df = pd.read_csv("data/gen_model_marginals_history.csv")


# In[33]:


#gen_model_history_df['lfs_26'] = L_dev_ci_marginals
#gen_model_history_df.head(2)


# In[34]:


#positive_class = dev_data_df.curated_dsh.values.sum()/dev_data_df.shape[0]
#plt.plot([0,1], [positive_class, positive_class], color='grey', 
#         linestyle='--', label='Baseline (AUC = {:0.2f})'.format(positive_class))

#for marginal, model_label in zip(gen_model_history_df.columns, ["17 LFs (AUC {:.2f})", "26 LFs (AUC {:.2f})"]):
#    precision, recall, threshold = precision_recall_curve(dev_data_labels, gen_model_history_df[marginal])
#    area = auc(recall, precision)
#    plt.plot(recall, precision, label=model_label.format(area))
#plt.title("Precision Recall Curve of Generative Models")
#plt.xlabel("Recall")
#plt.ylabel("Precision")
#plt.legend()


# In[35]:


#plt.plot([0,1], [0,1], linestyle='--', color='grey')
#for marginal, model_label in zip(gen_model_history_df.columns, ["17 LFs (AUC {:.2f})", "26 LFs (AUC {:.2f})"]):
#    fpr, tpr, threshold = roc_curve(dev_data_labels, gen_model_history_df[marginal])
#    area = auc(fpr, tpr)
#    plt.plot(fpr, tpr, label=model_label.format(area))
#plt.title("ROC Curve of Generative Models")
#plt.xlabel("False Positive Rate")
#plt.ylabel("True Positive Rate")
#plt.legend()


# In[36]:


#gen_model_history_df.to_csv("gen_model_marginals_history.csv", index=False)


# # F1 Score of Train Hand Labeled Set

# Looking at the small hand labeled training set we can see a pretty big spike in performance. In terms of f1 score the DA model has about a 0.25 increase in performance comapred to the CI model. 

# In[37]:


train_hand_labels = train_hand_df.curated_dsh.astype(int).tolist()


# In[38]:


#tp fp tn fn
indep_results = {}
for columns, models, name in zip(lfs_columns, indep_models, model_names):
    print(name)
    indep_results[name] = models.error_analysis(session, L_train_labeled[:, columns], train_hand_labels)


# In[39]:


dep_results = {}
for columns, models, name in zip(lfs_columns, dep_models, model_names):
    print(name)
    dep_results[name] = models.error_analysis(session, L_train_labeled[:, columns], train_hand_labels)


# In[40]:


train_hand_marginals_indep_df = pd.DataFrame(
    np.array([model.marginals(L_train_labeled[:, columns]) for model, columns in zip(indep_models, lfs_columns)]).T,
    columns=model_names
)

train_hand_marginals_dep_df = pd.DataFrame(
    np.array([model.marginals(L_train_labeled[:, columns]) for model, columns in zip(dep_models, lfs_columns)]).T,
    columns=model_names
)


# In[46]:


plt.figure(figsize=(10,6))
plt.plot([0,1], [0,1], linestyle='--', color='grey', label="Random (AUC = 0.50)")

for marginal, model_label in zip(train_hand_marginals_indep_df.columns, model_names):
    fpr, tpr, threshold = roc_curve(train_hand_labels, train_hand_marginals_dep_df[marginal])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model_label+" (AUC = {:0.2f})".format(area))
    
plt.title("ROC of INDEP Generative Models")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()


# In[47]:


plt.figure(figsize=(10,6))
plt.plot([0,1], [0,1], linestyle='--', color='grey', label="Random (AUC = 0.50)")

for marginal, model_label in zip(train_hand_marginals_dep_df.columns, model_names):
    fpr, tpr, threshold = roc_curve(train_hand_labels, train_hand_marginals_dep_df[marginal])
    area = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=model_label+" (AUC = {:0.2f})".format(area))
    
plt.title("ROC of DEP Generative Models")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()


# In[48]:


plt.figure(figsize=(10,6))
positive_class = dev_data_df.curated_dsh.values.sum()/dev_data_df.shape[0]
plt.plot([0,1], [positive_class, positive_class], color='grey', 
         linestyle='--', label='Baseline (AUC = {:0.2f})'.format(positive_class))

for marginal, model_label in zip(train_hand_marginals_indep_df.columns, model_names):
    precision, recall, threshold = precision_recall_curve(train_hand_labels, train_hand_marginals_indep_df[marginal])
    area = auc(recall, precision)
    plt.plot(recall, precision, label=model_label+" (AUC = {:0.2f})".format(area))
    
plt.title("Precision Recall Curve of INDEP Generative Models")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()


# In[44]:


L_train_labeled[:, lfs_columns[-1]].lf_stats(session, train_hand_df.curated_dsh.astype(int).tolist(), test_df.query("model=='DA'")["Learned Acc."])


# ## Individual Candidate Error Analysis

# Depending on which block of code is executed, the following block of code below will show which candidate sentence was incorrectly labeled. Right now the false negatives (fn) are being shown below but this could change to incorporate false positives (fp) as well.

# In[ ]:


from snorkel.viewer import SentenceNgramViewer

# NOTE: This if-then statement is only to avoid opening the viewer during automated testing of this notebook
# You should ignore this!
import os
if 'CI' not in os.environ:
    sv = SentenceNgramViewer(indep_results['CG_ALL'][1], session)
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


train_hand_marginals_indep_df.iloc[L_train_labeled.get_row_index(c)]


# ## Generate Excel File of Train Data

# Lastly we write out the generative model's output into a file. Reason for this will be used in the [next notebook](4.sentence-level-prediction.ipynb), where we aim to use a noise aware discriminator model to correct for the generative models' errors.

# In[ ]:


def make_sentence_df(lf_matrix, marginals, pair_df):
    rows = list()
    for i in tqdm_notebook(range(lf_matrix.shape[0])):
        row = OrderedDict()
        candidate = lf_matrix.get_candidate(session, i)
        row['candidate_id'] = candidate.id
        row['compound'] = candidate[0].get_span()
        row['gene'] = candidate[1].get_span()
        row['drugbank_id'] = candidate.Compound_cid
        row['entrez_gene_id'] = candidate.Gene_cid
        row['sentence'] = candidate.get_parent().text
        row['label'] = marginals[i]
        rows.append(row)
    sentence_df = pd.DataFrame(rows)
    sentence_df['entrez_gene_id'] = sentence_df.entrez_gene_id.astype(int)
    sentence_df = pd.merge(
        sentence_df,
        pair_df[["drugbank_id", "entrez_gene_id", "name", "gene_symbol"]],
        on=["drugbank_id", "entrez_gene_id"],
        how="left"
    )
    sentence_df = pd.concat([
        sentence_df,
        pd.DataFrame(lf_matrix.todense(), columns=list(LFS))
    ], axis='columns')
    return sentence_df


# In[ ]:


#pair_df = pd.read_csv("data/disease-gene-pairs-association.csv.xz", compression='xz')
pair_df = pd.read_csv("data/compound-gene-pairs-binds.csv")
pair_df.head(2)


# In[ ]:


train_sentence_df = make_sentence_df(L_train, train_marginals_indep, pair_df)
train_sentence_df.head(2)


# In[ ]:


writer = pd.ExcelWriter('data/compound_gene/sentence-labels.xlsx')
(train_sentence_df
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()


# ## Generate Excel File of Dev Data

# In[ ]:


dev_sentence_df = make_sentence_df(L_dev, L_dev_ci_marginals, pair_df)
dev_sentence_df.head(2)


# In[ ]:


writer = pd.ExcelWriter('data/compound_gene/sentence-labels-dev.xlsx')
(dev_sentence_df
    .sample(frac=1, random_state=100)
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()

