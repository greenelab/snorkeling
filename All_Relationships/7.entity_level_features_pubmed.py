
# coding: utf-8

# # Data Preparation for True Relationship Prediction

# After predicting the co-occurence of candidates on the sentence level, the next step is to predict whether a candidate is a true relationship or just occured by chance. Through out this notebook the main events involved here are calculating summary statistics and obtaining the LSTM marginal probabilities.

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

from collections import defaultdict
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
import scipy
from scipy.stats import fisher_exact
from scipy.special import logit

from sqlalchemy import and_

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


from snorkel.models import Candidate, candidate_subclass


# In[4]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# # Count the Number of Sentences for Each Candidate

# For this block of code we are cycling through each disease-gene candidate in the database and counting the number of unique sentences and unique abstracts containing the specific candidate. **NOTE**: This section will quite a few hours to cycle through the entire database.

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'pair_to_pmids = defaultdict(set)\npair_to_sentences = defaultdict(set)\noffset = 0\nchunk_size = 1e5\n\nwhile True:\n    cands = session.query(DiseaseGene).limit(chunk_size).offset(offset).all()\n    \n    if not cands:\n        break\n        \n    for candidate in cands:\n        pair = candidate.Disease_cid, candidate.Gene_cid\n        pair_to_sentences[pair].add(candidate[0].get_parent().id)\n        pair_to_pmids[pair].add(candidate[0].get_parent().document_id)\n\n    offset+= chunk_size')


# In[ ]:


candidate_df = pd.DataFrame(
    map(lambda x: [x[0], x[1], len(pair_to_sentences[x]), len(pair_to_pmids[x])], pair_to_sentences),
    columns=["disease_id", "gene_id", "sentence_count", "abstract_count"]
    )


# # Perform Fisher Exact Test on each Co-Occurence

# Here we want to perform the fisher exact test for each disease-gene co-occurence. A more detailed explanation [here](https://github.com/greenelab/snorkeling/issues/26).

# In[ ]:


def diffprop(obs):
    """
    `obs` must be a 2x2 numpy array.

    Returns:
    delta
        The difference in proportions
    ci
        The Wald 95% confidence interval for delta
    corrected_ci
        Yates continuity correction for the 95% confidence interval of delta.
    """
    n1, n2 = obs.sum(axis=1)
    prop1 = obs[0,0] / n1.astype(np.float64)
    prop2 = obs[1,0] / n2.astype(np.float64)
    delta = prop1 - prop2

    # Wald 95% confidence interval for delta
    se = np.sqrt(prop1*(1 - prop1)/n1 + prop2*(1 - prop2)/n2)
    ci = (delta - 1.96*se, delta + 1.96*se)

    # Yates continuity correction for confidence interval of delta
    correction = 0.5*(1/n1 + 1/n2)
    corrected_ci = (ci[0] - correction, ci[1] + correction)

    return delta, ci, corrected_ci


# In[ ]:


total = candidate_df["sentence_count"].sum()
odds = []
p_val = []
expected = []
lower_ci = []
epsilon = 1e-36

for disease, gene in tqdm.tqdm(zip(candidate_df["disease_id"],candidate_df["gene_id"])):
    cond_df = candidate_df.query("disease_id == @disease & gene_id == @gene")
    a = cond_df["sentence_count"].values[0] + 1
                                        
    cond_df = candidate_df.query("disease_id != @disease & gene_id == @gene")
    b = cond_df["sentence_count"].sum() + 1
    
    cond_df = candidate_df.query("disease_id == @disease & gene_id != @gene")
    c = cond_df["sentence_count"].sum() + 1
    
    cond_df = candidate_df.query("disease_id != @disease & gene_id != @gene")
    d = cond_df["sentence_count"].sum() + 1
    
    c_table = np.array([[a, b], [c, d]])
    
    # Gather confidence interval
    delta, ci, corrected_ci = diffprop(c_table)
    lower_ci.append(ci[0])
    
    # Gather corrected odds ratio and p_values
    odds_ratio, p_value = fisher_exact(c_table, alternative='greater')
    odds.append(odds_ratio)
    p_val.append(p_value + epsilon)
    
    total_disease = candidate_df[candidate_df["disease_id"] == disease]["sentence_count"].sum()
    total_gene = candidate_df[candidate_df["gene_id"] == gene]["sentence_count"].sum()
    expected.append((total_gene * total_disease)/float(total))

candidate_df["nlog10_p_value"] = (-1 * np.log10(p_val))
candidate_df["co_odds_ratio"] = odds
candidate_df["co_expected_sen_count"] = expected
candidate_df["delta_lower_ci"] = lower_ci


# In[ ]:


candidate_df.sort_values("nlog10_p_value", ascending=False).head(1000)


# # Combine Sentence Marginal Probabilities

# In this section we incorporate the marginal probabilites that are calculated from the bi-directional LSTM used in the [previous notebook](4.sentence-level-prediction.ipynb). For each sentence we grouped them by their disease-gene mention and report their marginal probabilites in different quantiles (0, 0.2, 0.4, 0.6, 0.8). Lastly we took the average of each sentence marginal to generate the "avg_marginal" column.

# In[13]:


candidate_df = pd.read_csv("data/disease_gene_summary_stats.csv")
candidate_df = candidate_df[[column for column in candidate_df.columns if "lstm" not in column]]
candidate_df = candidate_df.drop(["disease_name", "gene_name"], axis=1)
candidate_df.head(2)


# In[6]:


train_marginals_df = pd.read_table("data/training_set_marginals.tsv")
train_marginals_df.head(2)


# In[7]:


dev_marginals_df = pd.read_table("data/dev_set_marginals.tsv")
dev_marginals_df.head(2)


# In[8]:


candidate_marginals_df=(train_marginals_df
 .append(dev_marginals_df)
 .groupby(["disease_id", "gene_id"], as_index=False).mean()
)
candidate_marginals_df.head(2)


# In[9]:


dg_map = pd.read_csv("data/disease-gene-pairs-association.csv.xz")
dg_map.head(3)


# In[10]:


prior_df = pd.read_csv("data/observation-prior.csv")
prior_df['logit_prior_perm'] = prior_df.prior_perm.apply(logit)
prior_df.head(2)


# In[14]:


candidate_df = (candidate_df
 .merge(candidate_marginals_df)
 .merge(dg_map[["doid_id", "entrez_gene_id","split"]], 
        left_on=["disease_id", "gene_id"], 
        right_on=["doid_id", "entrez_gene_id"]
       )
 .drop(["doid_id", "entrez_gene_id"], axis='columns')
 .merge(prior_df[["disease_id", "gene_id", "logit_prior_perm"]], on=["disease_id", "gene_id"])
)
candidate_df.head(2)


# ## Save the data to a file

# In[15]:


candidate_df.to_csv("data/disease_gene_association_features.tsv", sep="\t", index=False)

