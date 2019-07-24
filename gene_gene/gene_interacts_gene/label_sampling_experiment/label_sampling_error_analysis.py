
# coding: utf-8

# # Using Labels from Different Relation Types to Predict Compound Binds Gene Sentences

# This notebook is designed to predict the compound binds gene (CbG) relation. The first step in this process is to label our train, dev, and test sentences (split = 6,7,8). We will label these sentences using all of our handcrafted label functions. The working hypothesis here is there are shared information between different relations, which in turn should aid in predicting the compound binds gene relation. After the labeling process, the next step is to train a generative model that will estimate the likelihood of the positive class ($\hat{Y}$) given our annotated label matrix. **Note**: This process doesn't involve any sentence context, so the only information used here are categorical output.

# ## Set up the environment

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import os
import pickle
import sys

sys.path.append(os.path.abspath('../../../modules'))

# Bayesian Optimization
from hyperopt import fmin, hp, tpe, Trials

from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, accuracy_score
from tqdm import tqdm_notebook


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


from snorkel.learning.pytorch.rnn.rnn_base import mark_sentence
from snorkel.learning.pytorch.rnn.utils import candidate_to_tokens
from snorkel.models import Candidate, candidate_subclass

from metal.analysis import lf_summary
from metal.label_model import LabelModel
from metal.utils import plusminus_to_categorical

from gensim.models import FastText
from gensim.models import KeyedVectors

from utils.notebook_utils.label_matrix_helper import label_candidates, get_auc_significant_stats
from utils.notebook_utils.dataframe_helper import load_candidate_dataframes, generate_results_df
from utils.notebook_utils.plot_helper import plot_curve, plot_label_matrix_heatmap
from utils.notebook_utils.train_model_helper import (
    train_baseline_model,
    run_random_additional_lfs
)

sys.path.append(os.path.abspath('../data/label_functions'))
sys.path.append(os.path.abspath('../../../disease_gene/disease_associates_gene/data/label_functions'))
sys.path.append(os.path.abspath('../../../compound_disease/compound_treats_disease/data/label_functions'))
sys.path.append(os.path.abspath('../../../compound_gene/compound_binds_gene/data/label_functions'))
from compound_gene_lf import CG_LFS
from disease_gene_lfs import DG_LFS
from compound_disease_lf import CD_LFS
from gene_gene_lf import GG_LFS


# In[4]:


GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])


# In[5]:


quick_load = True


# ## Load the Data and Label the  Sentences

# This block of code is designed to label the sentences using our label functions. All the sentences are located in our postgres database that is store locally on the lab machine. The labeling process is defined as follows: Given a candidate id, we use the sqlalchemy library to extract a candidate object. Using this object and we pass it through a series of label functions that will output a 1 (positive), -1 (negative) or 0 (abstain) depending on the rule set. Lastly we aggregate the output of these functions into a sparse matrix that the generative model will use. Since these steps are pretty linear, we parallelized this process using python's multithreading library. Despite the optimization, this process can still take about 3 hours to label a set of ~300000 sentences.

# In[6]:


total_candidates_df = pd.read_table("../dataset_statistics/results/all_gig_candidates.tsv.xz")
total_candidates_df.head(2)


# In[7]:


spreadsheet_names = {
    #'train': 'data/sentences/sentence_labels_train.xlsx',
    'dev': '../data/sentences/sentence_labels_dev.xlsx',
    'test': '../data/sentences/sentence_labels_test.xlsx'
}


# In[8]:


candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key], "curated_gig")
    for key in spreadsheet_names
}

for key in candidate_dfs:
    print("Size of {} set: {}".format(key, candidate_dfs[key].shape[0]))


# In[9]:


lfs = (
    list(CG_LFS["CbG"].values()) + 
    list(DG_LFS["DaG"].values())[7:37] + 
    list(CD_LFS["CtD"].values())[3:25] + 
    list(GG_LFS["GiG"].values())[9:37]
)
lf_names = (
    list(CG_LFS["CbG"].keys()) + 
    list(DG_LFS["DaG"].keys())[7:37] + 
    list(CD_LFS["CtD"].keys())[3:25] + 
    list(GG_LFS["GiG"].keys())[9:37]
)


# In[10]:


if not quick_load:
    label_matricies = {
        'train':label_candidates(
            session, 
            (
                total_candidates_df
                .query("split==3&gene_mention_count==2")
                .candidate_id
                .values
                .tolist()
            ),
            lfs, 
            lf_names,
            num_threads=10,
            batch_size=50000,
            multitask=False
        )
    }


# In[11]:


if not quick_load:
    label_matricies.update({
        key:label_candidates(
            session, 
            candidate_dfs[key]['candidate_id'].values.tolist(),
            lfs, 
            lf_names,
            num_threads=10,
            batch_size=50000,
            multitask=False
        )
        for key in candidate_dfs
    })


# In[12]:


# Save the label matricies to a file for future loading/error analysis
if not quick_load:
    (
        label_matricies['train']
        .sort_values("candidate_id")
        .to_csv("../data/label_matricies/train_sparse_matrix.tsv.xz", sep="\t", index=False)
    )
    (
        label_matricies['dev']
        .sort_values("candidate_id")
        .to_csv("../data/label_matricies/dev_sparse_matrix.tsv.xz", sep="\t", index=False)
    )
    (
        label_matricies['test']
        .sort_values("candidate_id")
        .to_csv("../data/label_matricies/test_sparse_matrix.tsv.xz", sep="\t", index=False)
    )
# Quick load the label matricies
else:
    label_destinations = {
        'train':"../data/label_matricies/train_sparse_matrix.tsv.xz",
        'dev':"../data/label_matricies/dev_sparse_matrix.tsv.xz",
        'test':"../data/label_matricies/test_sparse_matrix.tsv.xz"
    }
    label_matricies = {
        key:pd.read_table(label_destinations[key]).to_sparse()
        for key in label_destinations
    }


# In[13]:


# Important Note Snorkel Metal uses a different coding scheme
# than the label functions output. (2 for negative instead of -1).
# This step corrects this problem by converting -1s to 2

correct_L = plusminus_to_categorical(
    label_matricies['train']
    .sort_values("candidate_id")
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
    .astype(int)
)

correct_L_dev = plusminus_to_categorical(
    label_matricies['dev']
    .sort_values("candidate_id")
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
    .astype(int)
)

correct_L_test = plusminus_to_categorical(
    label_matricies['test']
    .sort_values("candidate_id")
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
    .astype(int)
)


# In[14]:


lf_summary(
    sparse.coo_matrix(
        correct_L
    )
    .tocsr(), 
    lf_names=lf_names
)


# In[15]:


lf_summary(
    sparse.coo_matrix(
        correct_L_dev
    )
    .tocsr(), 
    lf_names=lf_names, 
    Y=candidate_dfs['dev'].curated_gig.apply(lambda x: 1 if x> 0 else 2)
)


# # Train Baseline Model

# This block trains the baseline model (Distant Supervision of CbG Databases) that will be used as a reference to compare against.

# In[16]:


ds_start = 0
ds_end = 9
regularization_grid = pd.np.round(pd.np.linspace(0.01, 5, num=5), 2)


# In[17]:


dev_ds_grid, test_ds_grid = train_baseline_model(
    correct_L, correct_L_dev, candidate_dfs['dev'].curated_gig.values,
    correct_L_test, list(range(ds_start, ds_end)), regularization_grid,
    train_marginal_dir="data/random_sampling/GiG/marginals/"
)

dev_baseline_marginals = list(dev_ds_grid.values())[0][:,0]
test_baseline_marginals = list(test_ds_grid.values())[0][:,0]

dev_ds_grid = (
    generate_results_df(
        dev_ds_grid, 
        candidate_dfs['dev'].curated_gig.values
    )
    .reset_index()
    .rename(index=str, columns={0:"AUPRC", 1:"AUROC", "index":"l2_param"})
)

test_ds_grid = (
    generate_results_df(
        test_ds_grid, 
        candidate_dfs['test'].curated_gig.values
    )
    .reset_index()
    .rename(index=str, columns={0:"AUPRC", 1:"AUROC", "index":"l2_param"})
)


# In[18]:


best_param = dev_ds_grid.query("AUROC==AUROC.max()").l2_param.values[0]
dev_baseline=dev_ds_grid.query("l2_param==@best_param").to_dict('records')
dev_baseline[0].update({"num_lfs": 0})


# In[19]:


test_baseline=test_ds_grid.query("l2_param==@best_param").to_dict('records')
test_baseline[0].update({"num_lfs": 0})


# In[20]:


dev_baseline_marginals = list(zip(dev_baseline_marginals, candidate_dfs['dev'].curated_gig.values))
test_baseline_marginals = list(zip(test_baseline_marginals, candidate_dfs['test'].curated_gig.values))


# In[21]:


count_fraction_correct = lambda x: 1 if (x.marginals > 0.5 and x.label==1) or (x.marginals < 0.5 and x.label==0) else 0


# # Random Label Function Positive Direction

# This block is designed to determine how many label functions are needed to achieve decent results.

# In[22]:


num_of_samples = 50
regularization_grid = pd.np.round(pd.np.linspace(0.01, 5, num=5), 2)


# In[23]:


dev_gig_pos_df = pd.DataFrame(dev_baseline)
test_gig_pos_df = pd.DataFrame(test_baseline)

pd.np.random.seed(100)
frequency = [0.25, 0.5, 0.75, 1]

for p_success in frequency:
    
    dev_random_grid = {}
    test_random_grid = {}
    
    for sample in tqdm_notebook(range(num_of_samples)):
        random_lf = pd.np.random.binomial(1, p_success, (correct_L.shape[0], 1))
        new_L = pd.np.concatenate([correct_L[:, ds_start:ds_end], random_lf], axis=1)

        random_lf = pd.np.random.binomial(1, p_success, (correct_L_dev.shape[0], 1))
        new_L_dev = pd.np.concatenate([correct_L_dev[:, ds_start:ds_end], random_lf], axis=1)

        random_lf = pd.np.random.binomial(1, p_success, (correct_L_test.shape[0], 1))
        new_L_test = pd.np.concatenate([correct_L_test[:, ds_start:ds_end], random_lf], axis=1)

        new_dev_grid, new_test_grid = train_baseline_model(
            new_L, new_L_dev, candidate_dfs['dev'].curated_gig.values, 
            new_L_test, list(range(ds_start, ds_end+1)), regularization_grid,
            train_marginal_dir="data/random_sampling/GiG/marginals/"
        )
        dev_random_grid[sample] = new_dev_grid[list(new_dev_grid.keys())[0]]
        test_random_grid [sample] = new_test_grid[list(new_test_grid.keys())[0]]
    
    dev_gig_pos_df=(
        dev_gig_pos_df
        .append(
            generate_results_df(
                dev_random_grid, 
                candidate_dfs['dev'].curated_gig.values
            )
            .rename(index=str, columns={0: "AUPRC", 1: "AUROC"})
            .assign(num_lfs=p_success),
            sort=False
        )
        .reset_index()
        .drop("index", axis=1)
    )
    
    test_gig_pos_df=(
        test_gig_pos_df
        .append(
            generate_results_df(
                test_random_grid, 
                candidate_dfs['test'].curated_gig.values
            )
            .rename(index=str, columns={0: "AUPRC", 1: "AUROC"})
            .assign(num_lfs=p_success),
            sort=False
        )
        .reset_index()
        .drop("index", axis=1)
    )


# In[24]:


db_coverage = pd.Series(pd.np.any(new_L[:, ds_start:7], axis=1)).astype(int)
print(db_coverage.value_counts())
db_coverage.mean()

#absent_coverage = pd.Series(pd.np.any(new_L[:, 7:ds_end], axis=1)).astype(int)
#random_coverage = pd.Series(pd.np.any(new_L[:,9:10], axis=1)).astype(int)
#(absent_coverage+random_coverage).value_counts()


# ## Dev Set Performance (AUPRC, AUROC)

# In[25]:


fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
sns.pointplot(x="num_lfs", y="AUPRC", data=dev_gig_pos_df, ax=axs[0])
sns.pointplot(x="num_lfs", y="AUROC", data=dev_gig_pos_df, ax=axs[1])
axs[1].set_ylim([0.5,1])


# ## Test Set Performance (AUPRC, AUROC)

# In[26]:


fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
sns.pointplot(x="num_lfs", y="AUPRC", data=test_gig_pos_df, ax=axs[0])
sns.pointplot(x="num_lfs", y="AUROC", data=test_gig_pos_df, ax=axs[1])
axs[1].set_ylim([0.5,1])


# In[27]:


dev_gig_pos_df.to_csv(
    "results/error_analysis/pos_random_lf_dev_performance.tsv", 
    index=False, sep="\t", float_format="%.5g"
)

test_gig_pos_df.to_csv(
    "results/error_analysis/pos_random_lf_test_performance.tsv", 
    index=False, sep="\t", float_format="%.5g"
)


# # Random Label Function Negative Direction

# This block is designed to determine how many label functions are needed to achieve decent results.

# In[28]:


num_of_samples = 50
regularization_grid = pd.np.round(pd.np.linspace(0.01, 5, num=5), 2)


# In[29]:


dev_gig_neg_df = pd.DataFrame(dev_baseline)
test_gig_neg_df = pd.DataFrame(test_baseline)

#dev_gig_marginals_df = pd.DataFrame(dev_baseline_marginals, columns=["marginals", "label"]).assign(num_lfs=0)
#test_gig_marginals_df = pd.DataFrame(test_baseline_marginals, columns=["marginals", "label"]).assign(num_lfs=0)

pd.np.random.seed(100)
frequency = [0.25, 0.5, 0.75, 1]

for p_success in frequency:
    
    dev_random_grid = {}
    test_random_grid = {}
    
    for sample in tqdm_notebook(range(num_of_samples)):
        random_lf = pd.np.random.binomial(1, p_success, (correct_L.shape[0], 1)) * 2
        new_L = pd.np.concatenate([correct_L[:, ds_start:ds_end], random_lf], axis=1)

        random_lf = pd.np.random.binomial(1, p_success, (correct_L_dev.shape[0], 1)) * 2
        new_L_dev = pd.np.concatenate([correct_L_dev[:, ds_start:ds_end], random_lf], axis=1)

        random_lf = pd.np.random.binomial(1, p_success, (correct_L_test.shape[0], 1)) * 2
        new_L_test = pd.np.concatenate([correct_L_test[:, ds_start:ds_end], random_lf], axis=1)

        new_dev_grid, new_test_grid = train_baseline_model(
            new_L, new_L_dev, candidate_dfs['dev'].curated_gig.values, 
            new_L_test, list(range(ds_start, ds_end+1)), regularization_grid,
            train_marginal_dir="data/random_sampling/GiG/marginals/"
        )
        dev_random_grid[sample] = new_dev_grid[list(new_dev_grid.keys())[0]]
        test_random_grid [sample] = new_test_grid[list(new_test_grid.keys())[0]]
    
    dev_gig_neg_df=(
        dev_gig_neg_df
        .append(
            generate_results_df(
                dev_random_grid, 
                candidate_dfs['dev'].curated_gig.values
            )
            .rename(index=str, columns={0: "AUPRC", 1: "AUROC"})
            .assign(num_lfs=p_success),
            sort=False
        )
        .reset_index()
        .drop("index", axis=1)
    )
    
    test_gig_neg_df=(
        test_gig_neg_df
        .append(
            generate_results_df(
                test_random_grid, 
                candidate_dfs['test'].curated_gig.values
            )
            .rename(index=str, columns={0: "AUPRC", 1: "AUROC"})
            .assign(num_lfs=p_success),
            sort=False
        )
        .reset_index()
        .drop("index", axis=1)
    )


# ## Dev Set Performance (AUPRC, AUROC)

# In[30]:


fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
sns.pointplot(x="num_lfs", y="AUPRC", data=dev_gig_neg_df, ax=axs[0])
sns.pointplot(x="num_lfs", y="AUROC", data=dev_gig_neg_df, ax=axs[1])
axs[1].set_ylim([0.5,1])


# ## Test Set Performance (AUPRC, AUROC)

# In[31]:


fig, axs = plt.subplots(ncols=2, figsize=(10, 5))
sns.pointplot(x="num_lfs", y="AUPRC", data=test_gig_neg_df, ax=axs[0])
sns.pointplot(x="num_lfs", y="AUROC", data=test_gig_neg_df, ax=axs[1])
axs[1].set_ylim([0.5,1])


# In[32]:


dev_gig_neg_df.to_csv(
    "results/error_analysis/neg_random_lf_dev_performance.tsv", 
    index=False, sep="\t", float_format="%.5g"
)

test_gig_neg_df.to_csv(
    "results/error_analysis/neg_random_lf_test_performance.tsv", 
    index=False, sep="\t", float_format="%.5g"
)

