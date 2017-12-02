
# coding: utf-8

# # Calculate Prior Probability of Edge Types

# This notebook calculates the prior probabiltity of an edge type through permutation. This notebook will be used for various edge types as this project progresses.

# In[1]:

from collections import defaultdict
import itertools
import statistics
import pandas as pd

import tqdm
from hetio.permute import permute_pair_list
get_ipython().magic(u'matplotlib inline')


# In[2]:

hetnet_df = pd.read_csv('hetnet_dg_kb.csv')


# In[3]:

disease_degree = dict(hetnet_df["disease_id"].value_counts())
gene_degree = dict(hetnet_df["gene_id"].value_counts())


# In[4]:

association_edge = defaultdict(set)
association_row = list()

for (disease, d_degree), (gene, g_degree) in tqdm.tqdm(itertools.product(disease_degree.items(), gene_degree.items())):
    association_row.append((disease, gene, d_degree, g_degree))
    association_edge[(d_degree, g_degree)].add((disease, gene))

pair_df = pd.DataFrame(association_row, columns=["disease_id", "gene_id", "disease_associates", "gene_associates"])
pair_df.head(10)


# In[5]:

associations = list(zip(hetnet_df["disease_id"], hetnet_df["gene_id"]))
print(len(associations))


# In[6]:

# Burn In
pair_list, stats = permute_pair_list(associations, multiplier=10)
burnin_stats = pd.DataFrame(stats)


# In[7]:

burnin_stats


# In[8]:

burnin_stats["unchanged"].plot()


# In[9]:

# Burnin Stats
multiplier = 3


# In[10]:

# calculate the total number of permutations
# divide the total number by half to prevent memory issues
n_perm = hetnet_df["disease_id"].nunique() * hetnet_df["gene_id"].nunique()
n_perm = int(n_perm * 0.5)


# In[11]:

get_ipython().run_cell_magic(u'time', u'', u'edges_to_prob = {x: list() for x in association_edge}\n\nfor i in tqdm.tqdm(range(n_perm)):\n    pair_list, stats = permute_pair_list(pair_list, multiplier=multiplier, seed=i)\n    \n    pair_set = set(pair_list)\n    for degree, probs in edges_to_prob.items():\n        edges = association_edge[degree]\n        probs.append(len(edges & pair_set) / len(edges))')


# In[12]:

rows = []

for (d_deg, g_deg), probs in tqdm.tqdm(edges_to_prob.items()):
    mean = statistics.mean(probs)
    std_error = statistics.stdev(probs) / len(probs) ** 0.5
    rows.append((d_deg, g_deg, mean, std_error))
    
perm_df = pd.DataFrame(rows, columns=['disease_associates', 'gene_associates', 'prior_perm', 'prior_perm_stderr'])
perm_df.head(10)


# In[13]:

# Add unpermuted treatment prevalence columns
rows = list()
association_set = set(associations)

for (d_deg, g_deg), edges in association_edge.items():
    n_associations = len(edges & association_set)
    rows.append((d_deg, g_deg, n_associations, len(edges)))
degree_prior_df = pd.DataFrame(rows, columns=['disease_associates', 'gene_associates', 'n_associations', 'n_possible'])
degree_prior_df = perm_df.merge(degree_prior_df)
degree_prior_df = degree_prior_df.sort_values(['disease_associates', 'gene_associates'], ascending=False)


# In[14]:

degree_prior_df.head(5)


# In[15]:

degree_prior_df.to_csv("degree-prior.csv", index=False, float_format='%.6g')


# In[16]:

obs_pair_df = pair_df.merge(perm_df)
obs_pair_df.head(10)


# In[17]:

obs_pair_df.to_csv("observation-prior.csv", index=False, float_format='%.6g')

