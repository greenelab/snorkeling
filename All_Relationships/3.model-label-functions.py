
# coding: utf-8

# # Train the Generative Model for Candidate Labeling

# This notebook is designed to run a generative model that snorkel uses to probabilistically label each candidate. (1 for positive label and -1 for negative label). Using this generative model, we will test the hypothesis: **modeling correlation structure between label functions provides better precision and recall than the conditionally independent model.**

# ## MUST RUN AT THE START OF EVERYTHING

# Import the necessary modules and set up the database for database operations.

# In[ ]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import os
from tqdm import tqdm_notebook

import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score


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
from snorkel.annotations import load_gold_labels
from snorkel.annotations import FeatureAnnotator, LabelAnnotator, save_marginals
from snorkel.learning import GenerativeModel
from snorkel.learning.structure import DependencySelector
from snorkel.learning.utils import MentionScorer
from snorkel.models import Candidate, FeatureKey, candidate_subclass, Label
from snorkel.utils import get_as_dict


# In[ ]:


from utils.compound_gene_lf import CG_LFS
from utils.disease_gene_lf import DG_LFS
from utils.notebook_utils.plot_helper import *


# In[ ]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])


# # Load preprocessed data 

# This code will load the corresponding label matricies that were generated in the previous notebook ([Notebook 2](2.data-labeler.ipynb)). This notebook has three matricies which are broken down as follows:
# 
# |Dataset|Size|Description|
# |:-----|-----|:-----|
# |L_train|50,000|Randomly sampled from our 2,700,000 training set|
# |L_dev|10,000|Randomly sampled from our 700,000 dev set. Only 200 have been hand labeled|
# |L_train_labeled|919|Have been hand labled from training set and is separate from (L_train).|

# In[ ]:


spreadsheet_names = {
    'train': 'data/compound_disease/sentence_labels_train.xlsx',
    'train_hand_label': 'data/compound_disease/sentence_labels_train_dev.xlsx',
    'dev': 'data/compound_disease/sentence_labels_dev.xlsx'
}


# In[ ]:


train_df = pd.read_excel(spreadsheet_names['train'])
train_ids = train_df.candidate_id.astype(int).tolist()
print("Train Data Size: {}".format(len(train_ids)))


# In[ ]:


dev_df = pd.read_excel(spreadsheet_names['dev'])
dev_df = dev_df[dev_df.curated_dsh.notnull()].sort_values("candidate_id")
dev_ids = list(map(int, dev_df.candidate_id.values))
print("Total Hand Labeled Dev Sentences: {}".format(len(dev_ids)))


# In[ ]:


train_hand_df = pd.read_excel(spreadsheet_names['train_hand_label'])
train_hand_df = train_hand_df[train_hand_df.curated_dsh.notnull()]
train_hand_label_ids = train_hand_df.candidate_id.astype(int).tolist()
print("Total Hand Labeled Train Sentences: {}".format(len(train_hand_label_ids)))


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'labeler = LabelAnnotator(lfs=[])\n\n# Only grab candidates that have labels\ncids = session.query(Candidate.id).filter(Candidate.id.in_(train_ids))\nL_train = labeler.load_matrix(session, cids_query=cids)\n\ncids = session.query(Candidate.id).filter(Candidate.id.in_(dev_ids))\nL_dev = labeler.load_matrix(session, cids_query=cids)\n\ncids = session.query(Candidate.id).filter(Candidate.id.in_(train_hand_label_ids))\nL_train_hand_label = labeler.load_matrix(session, cids_query=cids)')


# In[ ]:


print("Total Number of Label Functions: {}".format(L_train.shape[1]))


# # Train the Generative Model

# Here is the first step in classifying candidate sentences. We train a generative model to probabilistically label each training sentence. This means the model assigns a probability to each sentence indicating whether or not it mentions a given relatinoship (> 0.5 if yes, 0.5 < if no). The generative model snorkel uses is a [factor graph](http://deepdive.stanford.edu/assets/factor_graph.pdf) and further information on this model can be found in their paper [here](https://arxiv.org/abs/1711.10160).
# 
# The following code below trains two different generative models. One model follows the assumption that each label function is independent of each other, while the other model assumes there are dependancies between each function (e.g. $L_{1}$ correlates with $L_{2}$).

# In[ ]:


cg_db = get_columns(session, L_train, CG_LFS, "CbG_DB")
cg_text = get_columns(session, L_train, CG_LFS, "CbG_TEXT")
dg_text = get_columns(session, L_train, DG_LFS, "DaG_TEXT")


# In[ ]:


# This block defines a list of label function columns defined above
lfs_columns = [
    cg_text
]

# This block specifies the labels for the above label function columns
model_names = [
    "CbG_TEXT"
]


# In[ ]:


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


# In[ ]:


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

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'train_marginals_indep_df = create_marginal_df(L_train, indep_models, \n                                              lfs_columns,model_names, \n                                              train_df.candidate_id.values)\n\ntrain_marginals_dep_df = create_marginal_df(L_train, dep_models,\n                                              lfs_columns, model_names,\n                                              train_df.candidate_id.values)')


# In[ ]:


plot_cand_histogram(model_names, lf_columns, train_marginals_indep_df,
                    "CI Training Set Marginals", "Probability of Positive Class")


# In[ ]:


plot_cand_histogram(model_names, lf_columns, train_marginals_dep_df,
                    "CI Training Set Marginals", "Probability of Positive Class")


# # Training Set Statistics

# Taking a closer look into the training set predictions, we can see how each label function individually performed. The two dataframes below contain the follwoing information: number of candidate sentences a label function has labeled (coverage), number of candidate sentences a label function agreed with another label function (overlaps), number of candidates a label function disagreed with another label function (conflicts), and lastly, the accuracy each label function has after training the generative model (Learned Acc).

# In[ ]:


# Generate Statistics of Generative Model
indep_learned_stats_df = indep_models[-1].learned_lf_stats()
learned_stats_df = dep_models[-1].learned_lf_stats()


# In[ ]:


indep_results_df = L_train[:, lfs_columns[-1]].lf_stats(session, est_accs=indep_learned_stats_df['Accuracy'])
indep_results_df


# In[ ]:


dep_results_df = L_train[:, lfs_columns[-1]].lf_stats(session, est_accs=learned_stats_df['Accuracy'])
dep_results_df


# The following bar charts below depict the weights the generative model assigns to each label function. The conditional independent model relies heavily on LF_HETNET_ABSENT and LF_NO_CONCLUSION, while the dependancy aware model relies more on the database-backed label functions. Ultimately, the DA model emphasizes more postive labels compared to the CI model. 

# In[ ]:


test_df = pd.concat([
    results_df[["Learned Acc."]].assign(model="DA"),
    indep_results_df[["Learned Acc."]].assign(model="CI"), 
])
test_df = test_df.reset_index()
test_df.head(2)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,11))
sns.barplot(ax=ax,y="index", x="Learned Acc.", hue="model", data=test_df, palette=sns.color_palette("muted"))


# ## F1 Score of Dev Set

# Moving from the training set, we now can look at how well these models can predict our small dev set. Looking at the chart below, the conditionally independent model doesn't perform well compared to the dependency aware model. In terms of f1 score there is about a .2 difference, which provides evidence towards the dependency model performing better.

# In[ ]:


indep_results = {}
for columns, models, name in zip(lfs_columns, indep_models, model_names):
    print(name)
    indep_results[name] = models.error_analysis(session, L_dev[:, columns], dev_data_labels)


# In[ ]:


dep_results = {}
for columns, models, name in zip(lfs_columns, dep_models, model_names):
    print(name)
    dep_results[name] = models.error_analysis(session, L_dev[:, columns], dev_data_labels)


# In[ ]:


dev_marginals_indep_df = create_marginal_df(L_dev, indep_models,
                                            lfs_columns, model_names, 
                                            dev_df.candiadte_id.values)

dev_marginals_dep_df = create_marginal_df(L_dev, dep_models,
                                            lfs_columns, model_names, 
                                            dev_df.candiadte_id.values)


# In[ ]:


plot_roc_curve(dev_marginals_indep_df, dev_df, model_names, "INDEP Generative Model ROC")


# In[ ]:


plot_roc_curve(dev_marginals_dep_df, dev_df, model_names, "DEP Generative Model ROC")


# In[ ]:


plot_pr_curve(dev_marginals_indep_df, dev_df.curated_dsh.values, model_names, "INDEP Generative MOdel PR Curve")


# In[ ]:


plot_pr_curve(dev_marginals_dep_df, dev_df.curated_dsh.values, model_names, "DEP Generative MOdel PR Curve")


# In[ ]:


L_dev[:, lfs_columns[-1]].lf_stats(session, dev_data_labels, test_df.query("model=='CI'")["Learned Acc."])


# # F1 Score of Train Hand Labeled Set

# Looking at the small hand labeled training set we can see a pretty big spike in performance. In terms of f1 score the DA model has about a 0.25 increase in performance comapred to the CI model. 

# In[ ]:


train_hand_labels = train_hand_df.curated_dsh.astype(int).tolist()


# In[ ]:


#tp fp tn fn
indep_results = {}
for columns, models, name in zip(lfs_columns, indep_models, model_names):
    print(name)
    indep_results[name] = models.error_analysis(session, L_train_labeled[:, columns], train_hand_labels)


# In[ ]:


dep_results = {}
for columns, models, name in zip(lfs_columns, dep_models, model_names):
    print(name)
    dep_results[name] = models.error_analysis(session, L_train_labeled[:, columns], train_hand_labels)


# In[ ]:


train_hand_marginals_indep_df = create_marginal_df(
    L_train_hand_label, indep_models,
    lfs_columns, model_names, 
    train_hand_df.candiadte_id.values
)

train_hand_marginals_dep_df = create_marginal_df(
    L_train_hand_label, dep_models,
    lfs_columns, model_names, 
    train_hand_df.candiadte_id.values
)


# In[ ]:


plot_roc_curve(train_hand_marginals_indep_df, train_hand_df.curated_dsh.values,
               model_names, "INDEP Generative Model ROC")


# In[ ]:


plot_roc_curve(train_hand_marginals_dep_df, train_hand_df.curated_dsh.values,
               model_names, "DEP Generative Model ROC")


# In[ ]:


plot_pr_curve(train_hand_marginals_dep_df, train_hand_df.curated_dsh.values,
              model_names, "DEP Generative MOdel PR Curve")


# In[ ]:


L_train_labeled[:, lfs_columns[-1]].lf_stats(session, train_hand_df.curated_dsh.astype(int).tolist(), test_df.query("model=='DA'")["Learned Acc."])


# ## Label Function and Datasize Experiment

# In[ ]:


gen_model_history_df = pd.read_csv(
    "data/disease_gene/disease_associates_gene/"+
    "lf_data_size_experiment/marginal_results/gen_model_marginals_history.csv"
)


# In[ ]:


plot_roc_curve(gen_model_history_df, dev_data.curated_dsh, 
               gen_model_history_df.columns, "ROC Curve of Generative Models")


# In[ ]:


plot_pr_curve(gen_model_history_df, dev_data.curated_dsh, 
              gen_model_history_df.columns, "PR Curve of Generative Models")


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


# ## Write Marginals of best model to File for Next Notebook

# Lastly we write out the generative model's output into a file. Reason for this will be used in the [next notebook](4.sentence-level-prediction.ipynb), where we aim to use a noise aware discriminator model to correct for the generative models' errors.

# In[ ]:


best_model = ""
truncated_models = ["candidate_id", best_model]
train_marginals_df[truncated_models].to_csv("data/compound_disease/marginal_results/train_marginals.tsv", index=False, sep="\t")
dev_df[truncated_models].to_csv("data/compound_disease/marginal_results/dev_marginals.tsv", index=False, sep="\t")
train_hand_df[truncated_models].to_csv("data/compound_disease/marginal_results/train_hand_marginals.tsv", index=False, sep="\t")

