
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
import tqdm

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
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator
from utils.disease_gene_lf import LFS


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

# This code will load the corresponding label matricies that were generated in the previous notebook ([Notebook 2](2.data-labeler.ipynb)). This notebook has three matricies which are broken down as follows:
# 
# |Dataset|Size|Description|
# |:-----|-----|:-----|
# |L_train|50,000|Randomly sampled from our 2,700,000 training set|
# |L_dev|10,000|Randomly sampled from our 700,000 dev set. Only 200 have been hand labeled|
# |L_train_labeled|919|Have been hand labled from training set and is separate from (L_train).|

# In[6]:


train_candidate_ids = np.loadtxt('data/labeled_candidates.txt').astype(int).tolist()
train_candidate_ids[0:10]


# In[7]:


dev_data_df = pd.read_excel("data/sentence-labels-dev-hand-labeled.xlsx")
dev_data_df = dev_data_df[dev_data_df.curated_dsh.notnull()]
dev_candidate_ids = list(map(int, dev_data_df.candidate_id.values))
print("Total Hand Labeled Dev Sentences: {}".format(len(dev_candidate_ids)))


# In[8]:


get_ipython().run_cell_magic('time', '', 'labeler = LabelAnnotator(lfs=[])\n\n# Only grab candidates that have labels\ncids = session.query(Candidate.id).filter(Candidate.id.in_(train_candidate_ids))\nL_train = labeler.load_matrix(session, cids_query=cids)\n\ncids = session.query(Candidate.id).filter(Candidate.id.in_(dev_candidate_ids))\nL_dev = labeler.load_matrix(session,cids_query=cids)')


# In[9]:


sql = '''
SELECT candidate_id FROM gold_label
INNER JOIN Candidate ON Candidate.id=gold_label.candidate_id
WHERE Candidate.split=0;
'''
cids = session.query(Candidate.id).filter(Candidate.id.in_([x[0] for x in session.execute(sql)]))
L_train_labeled = labeler.load_matrix(session, cids_query=cids)
L_train_labeled_gold = load_gold_labels(session, annotator_name='danich1', cids_query=cids)


# In[10]:


print("Total Number of Hand Labeled Candidates: {}\n".format(L_train_labeled_gold.shape[0]))
print("Distribution of Labels:")
print(pd.DataFrame(L_train_labeled_gold.toarray(), columns=["labels"])["labels"].value_counts())


# In[11]:


print("Total Size of Train Data: {}".format(L_train.shape[0]))
print("Total Number of Label Functions: {}".format(L_train.shape[1]))


# # Train the Generative Model

# Here is the first step in classifying candidate sentences. We train a generative model to probabilistically label each training sentence. This means the model assigns a probability to each sentence indicating whether or not it mentions a given relatinoship (> 0.5 if yes, 0.5 < if no). The generative model snorkel uses is a [factor graph](http://deepdive.stanford.edu/assets/factor_graph.pdf) and further information on this model can be found in their paper [here](https://arxiv.org/abs/1711.10160).
# 
# The following code below trains two different generative models. One model follows the assumption that each label function is independent of each other, while the other model assumes there are dependancies between each function (e.g. $L_{1}$ correlates with $L_{2}$).

# In[12]:


get_ipython().run_cell_magic('time', '', '#Conditionally independent Generative Model\nindep_gen_model = GenerativeModel()\nindep_gen_model.train(\n    L_train,\n    epochs=30,\n    decay=0.95,\n    step_size=0.1 / L_train.shape[0],\n    reg_param=1e-6,\n    threads=50,\n)')


# In[13]:


# select the dependancies from the label matrix
ds = DependencySelector()
deps = ds.select(L_train, threshold=0.1)
len(deps)


# In[14]:


get_ipython().run_cell_magic('time', '', '# Model each label function and the underlying correlation structure\ngen_model = GenerativeModel(lf_propensity=True)\ngen_model.train(\n    L_train,\n    epochs=30,\n    decay=0.95,\n    step_size=0.1 / L_train.shape[0],\n    reg_param=1e-6,\n    threads=50,\n    deps=deps\n)')


# # Generative Model Statistics

# Now that both models have been trained, the next step is to generate some statistics about each model. The two histograms below show a difference between both models' output. The conditionally independent model (CI) predicts more negative candidates compared to the dependancy aware model (DA).

# In[15]:


# Generate Statistics of Generative Model
indep_learned_stats_df = indep_gen_model.learned_lf_stats()
learned_stats_df = gen_model.learned_lf_stats()


# In[16]:


get_ipython().run_cell_magic('time', '', 'train_marginals_indep = indep_gen_model.marginals(L_train)\ntrain_marginals = gen_model.marginals(L_train)')


# In[17]:


plt.hist(train_marginals_indep, bins=20)
plt.title("CI Training Set Marginals")
plt.ylabel("Frequency")
plt.xlabel("Probability of Positive Class")
plt.show()


# In[18]:


plt.hist(train_marginals, bins=20)
plt.title("DA Training Set Marginals")
plt.ylabel("Frequency")
plt.xlabel("Probability of Positive Class")
plt.show()


# # Training Set Statistics

# Taking a closer look into the training set predictions, we can see how each label function individually performed. The two dataframes below contain the follwoing information: number of candidate sentences a label function has labeled (coverage), number of candidate sentences a label function agreed with another label function (overlaps), number of candidates a label function disagreed with another label function (conflicts), and lastly, the accuracy each label function has after training the generative model (Learned Acc).

# In[19]:


indep_results_df = L_train.lf_stats(session, est_accs=indep_learned_stats_df['Accuracy'])
indep_results_df.head(2)


# In[20]:


results_df = L_train.lf_stats(session, est_accs=learned_stats_df['Accuracy'])
results_df.head(2)


# The following bar charts below depict the weights the generative model assigns to each label function. The conditional independent model relies heavily on two negative functions, while the dependancy aware model has similar characteristics. Both LF_HETNET_ABSENT AND LF_NO_CONCLUIONS have the highest weight while the distribution of positive functions between both models differs. 

# In[21]:


test_df = pd.concat([
    results_df[["Learned Acc."]].assign(model="DA"),
    indep_results_df[["Learned Acc."]].assign(model="CI"), 
])
test_df = test_df.reset_index()
test_df.head(2)


# In[22]:


fig, ax = plt.subplots(figsize=(9,7))
sns.barplot(ax=ax,y="index", x="Learned Acc.", hue="model", data=test_df, palette=sns.color_palette("muted"))


# ## F1 Score of Dev Set

# Moving from the training set, we now can look at how well these models can predict our small dev set. Looking at the chart below, the conditionally independent model doesn't perform well compared to the dependency aware model. In terms of f1 score there is about a .2 difference, which provides evidence towards the dependency model performing better.

# In[23]:


_ = indep_gen_model.error_analysis(session, L_dev, dev_data_df.curated_dsh.apply(lambda x: -1 if x==0 else x).values)


# In[24]:


tp, fp, tn, fn = gen_model.error_analysis(session, L_dev, dev_data_df.curated_dsh.apply(lambda x: -1 if x==0 else x).values)


# # F1 Score of Train Hand Labeled Set

# Looking at the small hand labeled training set we can see a pretty big spike in performance. In terms of f1 score the DA model has about a 0.3 increase in performance comapred to the CI model. 

# In[25]:


_ = indep_gen_model.error_analysis(session, L_train_labeled, L_train_labeled_gold)


# In[26]:


tp, fp, tn, fn = gen_model.error_analysis(session, L_train_labeled, L_train_labeled_gold)


# ## Individual Candidate Error Analysis

# Depending on which block of code is executed, the following block of code below will show which candidate sentence was incorrectly labeled. Right now the false negatives (fn) are being shown below but this could change to incorporate false positives (fp) as well.

# In[27]:


from snorkel.viewer import SentenceNgramViewer

# NOTE: This if-then statement is only to avoid opening the viewer during automated testing of this notebook
# You should ignore this!
import os
if 'CI' not in os.environ:
    sv = SentenceNgramViewer(fn, session)
else:
    sv = None


# In[28]:


sv


# In[29]:


c = sv.get_selected() if sv else list(fp.union(fn))[0]
c


# In[30]:


c.labels


# In[31]:


c.id


# ## Generate Excel File of Train Data

# Lastly we write out the generative model's output into a file. Reason for this will be used in the [next notebook](4.sentence-level-prediction.ipynb), where we aim to use a noise aware discriminator model to correct for the generative models' errors.

# In[32]:


def make_sentence_df(lf_matrix, marginals, pair_df):
    rows = list()
    for i in tqdm.tqdm(range(lf_matrix.shape[0])):
        row = OrderedDict()
        candidate = lf_matrix.get_candidate(session, i)
        row['candidate_id'] = candidate.id
        row['disease'] = candidate[0].get_span()
        row['gene'] = candidate[1].get_span()
        row['doid_id'] = candidate.Disease_cid
        row['entrez_gene_id'] = candidate.Gene_cid
        row['sentence'] = candidate.get_parent().text
        row['label'] = marginals[i]
        rows.append(row)
    sentence_df = pd.DataFrame(rows)
    sentence_df['entrez_gene_id'] = sentence_df.entrez_gene_id.astype(int)
    sentence_df = pd.merge(
        sentence_df,
        pair_df[["doid_id", "entrez_gene_id", "doid_name", "gene_symbol"]],
        on=["doid_id", "entrez_gene_id"],
        how="left"
    )
    sentence_df = pd.concat([
        sentence_df,
        pd.DataFrame(lf_matrix.todense(), columns=list(LFS))
    ], axis='columns')
    return sentence_df


# In[33]:


pair_df = pd.read_csv("data/disease-gene-pairs-association.csv.xz", compression='xz')
pair_df.head(2)


# In[34]:


train_sentence_df = make_sentence_df(L_train, train_marginals, pair_df)
train_sentence_df.head(2)


# In[35]:


writer = pd.ExcelWriter('data/sentence-labels.xlsx')
(train_sentence_df
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()


# ## Generate Excel File of Dev Data

# In[ ]:


dev_sentence_df = make_sentence_df(L_dev, dev_marginals, pair_df)
dev_sentence_df.head(2)


# In[ ]:


writer = pd.ExcelWriter('data/sentence-labels-dev.xlsx')
(dev_sentence_df
    .sample(frac=1, random_state=100)
    .to_excel(writer, sheet_name='sentences', index=False)
)
if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)
writer.close()

