#!/usr/bin/env python
# coding: utf-8

# # Gene Interacts Gene Edge Prediction

# This notebook is designed to take the next step moving from predicted sentences to edge predictions. After training the discriminator model, each sentences contains a confidence score for the likelihood of mentioning a relationship. Multiple relationships contain multiple sentences, which makes establishing an edge unintuitive. Is taking the max score appropiate for determining existence of an edge? Does taking the mean of each relationship make more sense? The answer towards these questions are shown below.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import os
import sys
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import plotnine as p9
import seaborn as sns

sys.path.append(os.path.abspath('../../../modules'))

from utils.notebook_utils.dataframe_helper import mark_sentence


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


from snorkel.learning.pytorch.rnn.utils import candidate_to_tokens
from snorkel.models import Candidate, candidate_subclass


# In[4]:


GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])


# In[5]:


total_candidates_df = (
    pd
    .read_table("input/all_gig_candidates.tsv.xz")
    .sort_values("candidate_id")
)
total_candidates_df.head(2)


# In[6]:


sentence_prediction_df = (
    pd
    .read_table("input/all_predicted_gig_sentences.tsv.xz")
    .sort_values("candidate_id")
)
sentence_prediction_df.head(2)


# In[7]:


# DataFrame that combines likelihood scores with each candidate sentence
total_candidates_pred_df = (
    total_candidates_df[[
    "gene1_id", "gene1_name", 
    "gene2_id", "gene2_name", 
    "text", "hetionet",
    "candidate_id", "split"
    ]]
    .merge(sentence_prediction_df, on="candidate_id")
)

#total_candidates_pred_df.to_csv(
#    "output/combined_predicted_gig_sentences.tsv.xz", 
#    sep="\t", index=False, compression="xz"
#)

total_candidates_pred_df.head(2)


# In[8]:


# DataFrame that groups gene mentions together and takes
# the max, median and mean of each group
grouped_candidates_pred_df=(
    total_candidates_pred_df
    .groupby(["gene1_id", "gene2_id"], as_index=False)
    .agg({
        "pred": ['max', 'mean', 'median'], 
        'hetionet': 'max',
        "gene1_name": 'first',
        "gene2_name": 'first',
        "split": 'first'
    })
)
grouped_candidates_pred_df.head(2)


# In[9]:


grouped_candidates_pred_df.columns = [
    "_".join(col) 
    if col[1] != '' and col[0] not in ['hetionet', 'gene1_name', 'gene2_name', 'split'] else col[0] 
    for col in grouped_candidates_pred_df.columns.values
]
grouped_candidates_pred_df.head(2)


# In[10]:


grouped_candidates_pred_subet_df = (
    grouped_candidates_pred_df
    .query("split==5")
    .drop("split", axis=1)
)
grouped_candidates_pred_subet_df.head(2)


# In[11]:


grouped_candidates_pred_subet_df.hetionet.value_counts()


# # Best Sentence Representation Metric

# This section aims to answer the question: What metric (Mean, Max, Median) best predicts Hetionet Edges?

# In[12]:


performance_map = {}


# In[13]:


precision, recall, pr_threshold = precision_recall_curve(
    grouped_candidates_pred_subet_df.hetionet, 
    grouped_candidates_pred_subet_df.pred_max,
)

fpr, tpr, roc_threshold = roc_curve(
    grouped_candidates_pred_subet_df.hetionet, 
    grouped_candidates_pred_subet_df.pred_max,
)

performance_map['max'] = {
    "precision":precision, "recall":recall, 
    "pr_threshold":pr_threshold, "false_pos":fpr,
    "true_pos":tpr, "roc_threshold":roc_threshold, 
}


# In[14]:


precision, recall, pr_threshold = precision_recall_curve(
    grouped_candidates_pred_subet_df.hetionet, 
    grouped_candidates_pred_subet_df.pred_mean,
)

fpr, tpr, roc_threshold = roc_curve(
    grouped_candidates_pred_subet_df.hetionet, 
    grouped_candidates_pred_subet_df.pred_mean,
)

performance_map['mean'] = {
    "precision":precision, "recall":recall, 
    "pr_threshold":pr_threshold, "false_pos":fpr,
    "true_pos":tpr, "roc_threshold":roc_threshold, 
}


# In[15]:


precision, recall, pr_threshold = precision_recall_curve(
    grouped_candidates_pred_subet_df.hetionet, 
    grouped_candidates_pred_subet_df.pred_median,
)

fpr, tpr, roc_threshold = roc_curve(
    grouped_candidates_pred_subet_df.hetionet, 
    grouped_candidates_pred_subet_df.pred_median,
)

performance_map['median'] = {
    "precision":precision, "recall":recall, 
    "pr_threshold":pr_threshold, "false_pos":fpr,
    "true_pos":tpr, "roc_threshold":roc_threshold, 
}


# In[16]:


for key in performance_map:
    plt.plot(
        performance_map[key]['false_pos'], 
        performance_map[key]['true_pos'], 
        label=f"{key}:AUC ({auc(performance_map[key]['false_pos'], performance_map[key]['true_pos']):.3f})"
    )
plt.plot([0,1], [0,1], linestyle='--', color='black')
plt.legend()
plt.show()


# In[17]:


for key in performance_map:
    plt.plot(
        performance_map[key]['recall'], 
        performance_map[key]['precision'], 
        label=f"{key}:AUC ({auc(performance_map[key]['recall'], performance_map[key]['precision']):.3f})"
    )

plt.legend()
plt.show()


# # Optimal Cutoff Using PR-CURVE 

# In[18]:


threshold_df = (
    pd.DataFrame(
        list(
            zip(
                performance_map['max']['precision'], 
                performance_map['max']['recall'], 
                performance_map['max']['pr_threshold']
            )
        ),
        columns=["precision", "recall", "pr_threshold"]
    )
    .sort_values("precision", ascending=False)
)
threshold_df.head(2)


# In[19]:


#precision_thresholds = pd.np.linspace(0,1,num=5)
precision_thresholds = threshold_df.round(2).drop_duplicates("precision").precision.values

# Add the lowest precision rather than
# Keep it zero
precision_thresholds = (
    pd.np.where(
        precision_thresholds==0, 
        threshold_df.query("precision > 0").precision.min(), 
        precision_thresholds
    )
)

performance_records = []
for precision_cutoff in precision_thresholds:

    cutoff = (
        threshold_df
        .query("precision>=@precision_cutoff")
        .pr_threshold
        .min()
    )
    
    values_added = (
        grouped_candidates_pred_df
        .query("pred_max >= @cutoff")
        .hetionet
        .value_counts()
    )
    
    series_keys = list(values_added.keys())
    for key in series_keys:
        performance_records.append(
           {  
               "edges": values_added[key], 
               "in_hetionet": "Existing" if key == 1 else "Novel", 
               "precision": precision_cutoff,
               "sen_cutoff": cutoff
           }
        )
   
    
edges_added_df = (
    pd
    .DataFrame
    .from_records(performance_records)
)
edges_added_df.head(10)


# In[20]:


ax = sns.scatterplot(x="precision", y="edges", hue="in_hetionet", data=edges_added_df.sort_values("in_hetionet"))
ax.set(yscale="log")


# In[21]:


edges_added_df.to_csv("output/precision_gig_edges_added.tsv", index=False, sep="\t")


# # Total Recalled Edges

# How many edges of hetionet can we recall using a cutoff score of 0.5?

# In[22]:


def tag_sentence(x):
    candidates=(
        session
        .query(GeneGene)
        .filter(GeneGene.id.in_(x.candidate_id.astype(int).tolist()))
        .all()
    )
    tagged_sen=[
         " ".join(
             mark_sentence(
                candidate_to_tokens(cand), 
                [
                        (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                        (cand[1].get_word_start(), cand[1].get_word_end(), 2)
                ]
            )
         )
        for cand in candidates
    ]

    return tagged_sen


# In[23]:


gen_pred_df = (
    pd.read_csv("../label_sampling_experiment/results/GiG/marginals/train/28_sampled_train.tsv.xz", sep="\t")
    .iloc[:, [0,-1]]
    .append(
        pd.read_csv("../label_sampling_experiment/results/GiG/marginals/tune/28_sampled_dev.tsv", sep="\t")
        .iloc[:, [0,-1]]
    )
    .append(
        pd.read_csv("../label_sampling_experiment/results/GiG/marginals/test/28_sampled_test.tsv", sep="\t")
        .iloc[:, [0,-1]]
    )
)
gen_pred_df.columns = ["gen_pred", "candidate_id"]
gen_pred_df.head(2)


# In[24]:


(
    total_candidates_pred_df.iloc[
        total_candidates_pred_df
        .groupby(["gene1_id", "gene2_id"], as_index=False)
        .agg({
            "pred": 'idxmax'
        })
        .pred
    ]
    .merge(gen_pred_df, on=["candidate_id"])
    .assign(edge_type="GiG")
    .sort_values("pred", ascending=False)
    .query("gene1_name!=gene2_name")
    .head(10)
    .sort_values("candidate_id")
    .assign(text=lambda x: tag_sentence(x))
    .merge(total_candidates_df[["n_sentences", "candidate_id"]], on="candidate_id")
    .sort_values("pred", ascending=False)
    .assign(hetionet=lambda x: x.hetionet.apply(lambda x: "Existing" if x == 1 else "Novel"))
    [["edge_type", "gene1_name", "gene2_name", "gen_pred", "pred", "n_sentences", "hetionet", "text"]]
    .to_csv("output/top_ten_edge_predictions.tsv", sep="\t", index=False, float_format="%.3g")
)


# In[25]:


datarows = []
fpr, tpr, threshold = roc_curve(
    grouped_candidates_pred_df.hetionet.values, 
    grouped_candidates_pred_df.pred_max.values
)

fnr = 1 - tpr
optimal_threshold = threshold[pd.np.nanargmin(pd.np.absolute((fnr - fpr)))]

datarows.append({
    "recall":(
        grouped_candidates_pred_df
        .query("pred_max > @optimal_threshold")
        .hetionet
        .value_counts()[1] /
        grouped_candidates_pred_df
        .hetionet.
        value_counts()[1]
    ),
    "edges":(
        grouped_candidates_pred_df
        .query("pred_max > @optimal_threshold")
        .hetionet
        .value_counts()[1]
    ),
    "in_hetionet": "Existing",
    "total": int(grouped_candidates_pred_df.hetionet.value_counts()[1]),
    "relation":"GiG"
})
datarows.append({
    "edges":(
        grouped_candidates_pred_df
        .query("pred_max > @optimal_threshold")
        .hetionet
        .value_counts()[0]
    ),
    "in_hetionet": "Novel",
    "relation":"GiG"
})
edges_df = pd.DataFrame.from_records(datarows)
edges_df


# In[26]:


import math
g = (
    p9.ggplot(edges_df, p9.aes(x="relation", y="edges", fill="in_hetionet"))
    + p9.geom_col(position="dodge")
    + p9.geom_text(
        p9.aes(
            label=(
                edges_df
                .apply(
                    lambda x: 
                    f"{x['edges']} ({x['recall']*100:.0f}%)" 
                    if not math.isnan(x['recall']) else 
                    f"{x['edges']}",
                    axis=1
                )
            )
        ),
        position=p9.position_dodge(width=1),
        size=9,
        va="bottom"
    )
    + p9.scale_y_log10()
    + p9.theme(
        axis_text_y=p9.element_blank(),
        axis_ticks_major = p9.element_blank(),
        rect=p9.element_blank()
    )
)
print(g)

