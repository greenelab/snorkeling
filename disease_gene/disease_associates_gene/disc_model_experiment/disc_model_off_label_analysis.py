#!/usr/bin/env python
# coding: utf-8

# # Discriminator Model for Off Label Generative Model

# Given that transferring label functions doesn't improve performance, this notebook is designed to confirm that training a discriminator model on labels from a off-relation trained generative model would **fail** to increase predictive performance. Generative model here was trained using Compound treats Disease label functions and the discrimintator model is designed to predict Disease associates Gene sentences. Performance is reported in area under the receiver operating curve (AUROC) and area under the precision recall curve (AUPR).

# In[1]:


import glob
import os
import pandas as pd

import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve


# # Tune Set

# In[3]:


dev_labels = pd.read_csv("input/dag_dev_labels.tsv", sep="\t")
dev_labels.head()


# In[4]:


candidate_df = (
    pd.read_excel("../data/sentences/sentence_labels_dev.xlsx")
    .sort_values("candidate_id")
    .query("curated_dsh.notnull()")
)
candidate_df.head()


# In[5]:


gen_model_results_dev_df = pd.read_csv(
    "../label_sampling_experiment/results/CtD/results/dev_sampled_results.tsv", 
    sep="\t"
)


# In[7]:


disc_model_dict = {}
disc_model_dict['22'] = (
    pd.read_csv(f"input/neg/22/tune.tsv", sep="\t")
)


# In[8]:


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


# In[15]:


records = []
for sample in disc_model_dict:
        for column in disc_model_dict[sample].drop("candidate_id", axis=1).columns:
            aucs = get_au_performance(
                disc_model_dict[sample][column],
                candidate_df
                .query(f"candidate_id in {disc_model_dict[sample].candidate_id.values.tolist()}")
                .curated_dsh
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


# In[16]:


ax = sns.pointplot(
    x='lf_num', y='auroc',
    hue='model',
    data=dev_set_df,
    sd='ci'
)
for x_spot, lf_num in enumerate(sorted(dev_set_df.lf_num.unique())):
    print(lf_num)
    print(
        dev_set_df.query(f"lf_num=={lf_num}&model=='disc_model'").auroc.mean()-
        dev_set_df.query(f"lf_num=={lf_num}&model=='gen_model'").auroc.mean()
    )


# In[17]:


ax = sns.pointplot(
    x='lf_num', y='aupr',
    hue='model',
    data=dev_set_df,
    sd='ci'
)
for x_spot, lf_num in enumerate(sorted(dev_set_df.lf_num.unique())):
    print(lf_num)
    print(
        dev_set_df.query(f"lf_num=={lf_num}&model=='disc_model'").aupr.mean()-
        dev_set_df.query(f"lf_num=={lf_num}&model=='gen_model'").aupr.mean()
    )


# # Test Set

# In[18]:


test_labels = pd.read_csv("input/dag_test_labels.tsv", sep="\t")
test_labels.head()


# In[19]:


candidate_df = (
    pd.read_excel("../data/sentences/sentence_labels_test.xlsx")
    .sort_values("candidate_id")
    .query("curated_dsh.notnull()")
)
candidate_df.head()


# In[20]:


gen_model_results_test_df = pd.read_csv(
    "../label_sampling_experiment/results/CtD/results/test_sampled_results.tsv", 
    sep="\t"
)


# In[22]:


disc_model_dict = {}

disc_model_dict['22'] = (
        pd.read_csv(f"input/neg/22/test.tsv", sep="\t")
)


# In[23]:


records = []
for sample in disc_model_dict:
        for column in disc_model_dict[sample].drop("candidate_id", axis=1).columns:
            aucs = get_au_performance(
                disc_model_dict[sample][column],
                candidate_df
                .query(f"candidate_id in {disc_model_dict[sample].candidate_id.values.tolist()}")
                .curated_dsh
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


ax = sns.pointplot(
    x='lf_num', y='auroc',
    hue='model',
    data=test_set_df,
    sd='ci'
)
for x_spot, lf_num in enumerate(sorted(dev_set_df.lf_num.unique())):
    print(lf_num)
    print(
        test_set_df.query(f"lf_num=={lf_num}&model=='disc_model'").auroc.mean()-
        test_set_df.query(f"lf_num=={lf_num}&model=='gen_model'").auroc.mean()
    )


# In[25]:


ax = sns.pointplot(
    x='lf_num', y='aupr',
    hue='model',
    data=test_set_df,
    sd='ci'
)
for x_spot, lf_num in enumerate(sorted(dev_set_df.lf_num.unique())):
    print(lf_num)
    print(
        test_set_df.query(f"lf_num=={lf_num}&model=='disc_model'").aupr.mean()-
        test_set_df.query(f"lf_num=={lf_num}&model=='gen_model'").aupr.mean()
    )

