#!/usr/bin/env python
# coding: utf-8

# # Discriminator Model Performance

# This notebook is designed to analyze the discriminator model's performance. Once the generative model labels our data, the discriminator model takes those labels and improves on predictions. For this notebook we are using a generative model trained on Compound treats Disease label functions to predict Compound treats Disease sentences. Performance for each model is reported in area under the receiver operating curve (AUROC) and area under the precision recall curve (AUPR).

# In[1]:


import glob
import os
import pandas as pd

import plotnine as p9
import scipy.stats as ss
from sklearn.metrics import auc, precision_recall_curve, roc_curve, precision_recall_fscore_support


# # Tune Set

# ## Performance of Disc model vs Gen model for each Label Sample

# In[2]:


dev_labels = pd.read_csv("input/ctd_dev_labels.tsv", sep="\t")
dev_labels.head()


# In[3]:


candidate_df = (
    pd.read_excel("../data/sentences/sentence_labels_dev.xlsx")
    .sort_values("candidate_id")
    .query("curated_ctd.notnull()")
)
candidate_df.head()


# In[4]:


gen_model_results_dev_df = pd.read_csv(
    "../label_sampling_experiment/results/CtD/results/dev_sampled_results.tsv", 
    sep="\t"
)


# In[5]:


disc_model_dict = {}

for value in gen_model_results_dev_df.lf_num.unique():
        
    disc_model_dict[value] = (
        pd.read_csv(f"input/disc_model_run/{value}/tune.tsv", sep="\t")
    )


# In[6]:


def get_au_performance(predictions, gold_labels):
    fpr, tpr, _ = roc_curve(
        gold_labels,
        predictions
    )
    
    precision, recall, _ = precision_recall_curve(
        gold_labels,
        predictions
    )
    
    return auc(fpr, tpr), auc(recall, precision)


# In[7]:


records = []
for sample in disc_model_dict:
        for column in disc_model_dict[sample].drop("candidate_id", axis=1).columns:
            aucs = get_au_performance(
                disc_model_dict[sample][column], 
                candidate_df
                .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
                .curated_ctd
                .values
            )
            records.append({
                "model": "disc_model",
                "lf_num": int(sample),
                "auroc": aucs[0],
                "aupr": aucs[1]
            })

dev_set_df = (
    pd.DataFrame.from_records(records)
    .append(
        gen_model_results_dev_df
        .drop("lf_sample", axis=1)
        .assign(model="gen_model")
    )
)


# In[8]:


dev_set_stats_df = (
    dev_set_df
    .groupby(["lf_num", "model"])
    .agg({
    "auroc": ['mean', 'std'],
    "aupr": ['mean', 'std'],
    "lf_num": len
    })
    .reset_index()
    .fillna(0)
)
dev_set_stats_df.columns = [
    "_".join(col) 
    if col[1] != '' and col[0] not in ['hetionet', 'gene_symbol', 'doid_name', 'split'] else col[0] 
    for col in dev_set_stats_df.columns.values
]

critical_val = ss.norm.ppf(0.975)

dev_set_stats_df = (
    dev_set_stats_df
    .assign(
        **{
            'auroc_upper': lambda x: x.auroc_mean + (critical_val * x.auroc_std)/pd.np.sqrt(x.lf_num_len),
            'auroc_lower': lambda x: x.auroc_mean - (critical_val * x.auroc_std)/pd.np.sqrt(x.lf_num_len),
            'aupr_upper': lambda x: x.aupr_mean + (critical_val * x.aupr_std)/pd.np.sqrt(x.lf_num_len),
            'aupr_lower':lambda x: x.aupr_mean - (critical_val * x.aupr_std)/pd.np.sqrt(x.lf_num_len)
        }
    )
)
dev_set_stats_df


# In[9]:


(
    p9.ggplot(dev_set_stats_df, p9.aes(x="factor(lf_num)", y="auroc_mean", color="model"))
    + p9.geom_point()
    + p9.geom_line(p9.aes(group="model"))
    + p9.geom_errorbar(p9.aes(ymin="auroc_lower", ymax="auroc_upper", group="model"))
    + p9.theme_seaborn()
    + p9.labs(
        title= "CtD Tune Set AUROC",
        color="Model"
    )
    + p9.scale_color_manual({
        "disc_model": "blue",
        "gen_model": "orange"
    })
)


# In[10]:


(
    p9.ggplot(dev_set_stats_df, p9.aes(x="factor(lf_num)", y="aupr_mean", color="model"))
    + p9.geom_point()
    + p9.geom_line(p9.aes(group="model"))
    + p9.geom_errorbar(p9.aes(ymin="aupr_lower", ymax="aupr_upper", group="model"))
    + p9.theme_seaborn()
    + p9.labs(
        title= "CtD Tune Set AUPR",
        color="Model"
    )
    + p9.scale_color_manual({
        "disc_model": "blue",
        "gen_model": "orange"
    })
)


# In[11]:


dev_set_df.to_csv("output/dev_set_disc_performance.tsv", sep="\t", index=False)


# ##  Precision-Recall Improvement over Generative Model

# In[12]:


gen_predicton = (
    pd.read_csv(
        "../label_sampling_experiment/results/CtD/marginals/tune/22_sampled_dev.tsv",
        sep="\t"
    )
    .assign(candidate_id=candidate_df.candidate_id.values.tolist())
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .iloc[:,0]
)


# In[13]:


disc_precision, disc_recall, _ = precision_recall_curve(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values, 
    disc_model_dict[22]['0']
)

gen_precision, gen_recall, _ = precision_recall_curve(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values,
    gen_predicton
)


# In[14]:


pr_perform_df = (
    pd.DataFrame(
        {'precision':gen_precision, 'recall':gen_recall}
    )
    .assign(model='gen_model')
    .append(
        pd.DataFrame(
            {'precision':disc_precision, 'recall':disc_recall}
        )
        .assign(model='disc_model')
    )
)


# In[15]:


(
    p9.ggplot(pr_perform_df, p9.aes(x="recall", y="precision", color="factor(model)")) +
    p9.geom_point()+ 
    p9.geom_line() + 
    p9.labs(
        title= "Validation PR Curve",
        color="Model"
    )+
    p9.scale_color_discrete(l=.4)+
    p9.theme_seaborn()
)


# In[16]:


precision_recall_fscore_support(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values, 
    gen_predicton.apply(lambda x: 1 if x > 0.5 else 0),
    average='binary'
)


# In[17]:


precision_recall_fscore_support(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values, 
    disc_model_dict[22]['0'].apply(lambda x: 1 if x > 0.5 else 0),
    average='binary'
)


# In[18]:


pr_perform_df.to_csv("output/dev_set_pr_performance.tsv", sep="\t", index=False)


# # Test Set

# ## Performance of Disc model vs Gen model for each Label Sample

# In[19]:


test_labels = pd.read_csv("input/ctd_test_labels.tsv", sep="\t")
test_labels.head()


# In[20]:


candidate_df = (
    pd.read_excel("../data/sentences/sentence_labels_test.xlsx")
    .sort_values("candidate_id")
    .query("curated_ctd.notnull()")
)
candidate_df.head()


# In[21]:


gen_model_results_test_df = pd.read_csv(
    "../label_sampling_experiment/results/CtD/results/test_sampled_results.tsv", 
    sep="\t"
)


# In[22]:


disc_model_dict = {}

for value in gen_model_results_test_df.lf_num.unique():

    disc_model_dict[value] = (
        pd.read_csv(f"input/disc_model_run/{value}/test.tsv", sep="\t")
    )


# In[23]:


records = []
for sample in disc_model_dict:
        for column in disc_model_dict[sample].drop("candidate_id", axis=1).columns:
            aucs = get_au_performance(
                disc_model_dict[sample][column], 
                candidate_df
                .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
                .curated_ctd
                .values
            )
            records.append({
                "model": "disc_model",
                "lf_num": int(sample),
                "auroc": aucs[0],
                "aupr": aucs[1]
            })

test_set_df = (
    pd.DataFrame.from_records(records)
    .append(
        gen_model_results_test_df
        .drop("lf_sample", axis=1)
        .assign(model="gen_model")
    )
)


# In[24]:


test_set_stats_df = (
    test_set_df
    .groupby(["lf_num", "model"])
    .agg({
    "auroc": ['mean', 'std'],
    "aupr": ['mean', 'std'],
    "lf_num": len
    })
    .reset_index()
    .fillna(0)
)
test_set_stats_df.columns = [
    "_".join(col) 
    if col[1] != '' and col[0] not in ['hetionet', 'gene_symbol', 'doid_name', 'split'] else col[0] 
    for col in test_set_stats_df.columns.values
]

critical_val = ss.norm.ppf(0.975)

test_set_stats_df = (
    test_set_stats_df
    .assign(
        **{
            'auroc_upper': lambda x: x.auroc_mean + (critical_val * x.auroc_std)/pd.np.sqrt(x.lf_num_len),
            'auroc_lower': lambda x: x.auroc_mean - (critical_val * x.auroc_std)/pd.np.sqrt(x.lf_num_len),
            'aupr_upper': lambda x: x.aupr_mean + (critical_val * x.aupr_std)/pd.np.sqrt(x.lf_num_len),
            'aupr_lower':lambda x: x.aupr_mean - (critical_val * x.aupr_std)/pd.np.sqrt(x.lf_num_len)
        }
    )
)
test_set_stats_df


# In[25]:


(
    p9.ggplot(test_set_stats_df, p9.aes(x="factor(lf_num)", y="auroc_mean", color="model"))
    + p9.geom_point()
    + p9.geom_line(p9.aes(group="model"))
    + p9.geom_errorbar(p9.aes(ymin="auroc_lower", ymax="auroc_upper", group="model"))
    + p9.theme_seaborn()
    + p9.labs(
        title= "CtD Test Set AUROC",
        color="Model"
    )
    + p9.scale_color_manual({
        "disc_model": "blue",
        "gen_model": "orange"
    })
)


# In[26]:


(
    p9.ggplot(test_set_stats_df, p9.aes(x="factor(lf_num)", y="aupr_mean", color="model"))
    + p9.geom_point()
    + p9.geom_line(p9.aes(group="model"))
    + p9.geom_errorbar(p9.aes(ymin="aupr_lower", ymax="aupr_upper", group="model"))
    + p9.theme_seaborn()
    + p9.labs(
        title= "CtD Test Set AUPR",
        color="Model"
    )
    + p9.scale_color_manual({
        "disc_model": "blue",
        "gen_model": "orange"
    })
)


# In[27]:


test_set_df.to_csv("output/test_set_disc_performance.tsv", sep="\t", index=False)


# ##  Precision-Recall Improvement over Generative Model

# In[28]:


gen_predicton = (
    pd.read_csv(
        "../label_sampling_experiment/results/CtD/marginals/test/22_sampled_test.tsv",
        sep="\t"
    )
    .assign(candidate_id=candidate_df.candidate_id.values.tolist())
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .iloc[:,0]
)


# In[29]:


disc_precision, disc_recall, _ = precision_recall_curve(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values, 
    disc_model_dict[22]['0']
)

gen_precision, gen_recall, _ = precision_recall_curve(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values,
    gen_predicton
)


# In[30]:


pr_perform_df = (
    pd.DataFrame(
        {'precision':gen_precision, 'recall':gen_recall}
    )
    .assign(model='gen_model')
    .append(
        pd.DataFrame(
            {'precision':disc_precision, 'recall':disc_recall}
        )
        .assign(model='disc_model')
    )
)


# In[31]:


(
    p9.ggplot(pr_perform_df, p9.aes(x="recall", y="precision", color="factor(model)")) +
    p9.geom_point()+ 
    p9.geom_line() + 
    p9.labs(
        title= "Test PR Curve",
        color="Model"
    )+
    p9.scale_color_discrete(l=.4)+
    p9.theme_seaborn()
)


# In[32]:


precision_recall_fscore_support(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values, 
    gen_predicton.apply(lambda x: 1 if x > 0.5 else 0),
    average='binary'
)


# In[33]:


precision_recall_fscore_support(
    candidate_df
    .query(f"candidate_id in {disc_model_dict[value].candidate_id.values.tolist()}")
    .curated_ctd
    .values, 
    disc_model_dict[22]['0'].apply(lambda x: 1 if x > 0.5 else 0),
    average='binary'
)


# In[34]:


pr_perform_df.to_csv("output/test_set_pr_performance.tsv", sep="\t", index=False)

