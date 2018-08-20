
# coding: utf-8

# # Train/Test the True Relationship Model

# This notebook is design to predict DG relationships on the entity level. Here we are taking the input from this [paper](https://zenodo.org/record/1035500#.W3Hc-RgpBrk), prior probability notebook and the summary statistics notebook, then combinding it into a single dataset. From there we train a Ridge LR model and an elastic net LR model to make the final prediction.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
import re
from scipy.stats import fisher_exact
from scipy.special import logit
import scipy
from sqlalchemy import and_
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# # Incorporate [Russ's Data](https://zenodo.org/record/1035500#.W3Hc-RgpBrk) into our Entity Dataframe

# In[2]:


named_columns = ["pubmed_id", "sentence_num", "first_entity", 
                 "first_entity_location", "second_entity", "second_entity_location", 
                 "first_entity_string", "second_entity_string", "first_entity_db_id", 
                 "second_entity_db_id", "first_entity_type", "second_entity_type", 
                 "dependency_path", "sentence_string"
                ]
russ_df = pd.read_table("data/hierarchical_clustering/part-ii-dependency-paths-gene-disease-sorted-with-themes.txt", 
                        names=named_columns)
russ_df.head(2)


# In[3]:


disease_gene_df = pd.read_csv("data/disease_gene_summary_stats.csv")
disease_gene_df.head(2)


# In[4]:


disease_gene_russ_df = (
    disease_gene_df[[col for col in disease_gene_df.columns if "lstm" not in col]]
    .astype({'disease_name':str, 'gene_name':str})
    .merge(
        russ_df
        .astype({'first_entity':str, 'second_entity':str}),
        left_on=["gene_name", "disease_name"], right_on=["first_entity", "second_entity"], how='left')
)
(
    disease_gene_russ_df[
        ["disease_id", "disease_name", "gene_id", 
         "gene_name", "nlog10_p_value", "co_odds_ratio", 
         "co_expected_sen_count", "delta_lower_ci", "hetnet_labels", 
         "pubmed_id", "sentence_num", "dependency_path", "sentence_string"]
    ]
    .head(2)
)


# In[5]:


# fix the dependency path strings so they will match
def fix(x):
    x = re.sub('START_ENTITY', 'start_entity', x)
    x = re.sub('END_ENTITY', 'end_entity', x)
    return x


# In[6]:


disease_gene_russ_df.dependency_path = disease_gene_russ_df.dependency_path.astype(str).apply(fix)


# In[7]:


theme_dist_df = pd.read_table("data/hierarchical_clustering/part-i-gene-disease-path-theme-distributions.txt")
theme_dist_df.head(2)


# In[ ]:


# fix the dependency path strings so they will match
def fix(x):
    x = re.sub('START_ENTITY', 'start_entity', x)
    x = re.sub('END_ENTITY', 'end_entity', x)
    return x


# In[ ]:


disease_gene_russ_df.dependency_path = disease_gene_russ_df.dependency_path.astype(str).apply(fix)


# In[8]:


final_dg_df = (
    disease_gene_russ_df
    .astype({'dependency_path':str})
    .merge(theme_dist_df.astype({"path":str}), left_on='dependency_path', right_on='path', how='left')
)
final_dg_df.head(2)


# In[9]:


# Transform the nans into 0.0 and aggregate the other sentnece scores
final_dg_df[['U', 'U.ind','Ud', 'Ud.ind',
     'D', 'D.ind', 'J', 'J.ind',
     'Y', 'Y.ind', 'G', 'G.ind', 
     'Md', 'Md.ind', 'X', 'X.ind', 
     'L', 'L.ind']] = (
    final_dg_df.groupby(["disease_id", "gene_id"])[
        ['U', 'U.ind','Ud', 'Ud.ind',
         'D', 'D.ind', 'J', 'J.ind', 
         'Y', 'Y.ind', 'G', 'G.ind', 
         'Md', 'Md.ind', 'X', 'X.ind', 
         'L', 'L.ind']]
    .transform('sum')
)
final_dg_df.head(2)


# In[10]:


(
    final_dg_df
    .drop_duplicates(["disease_name", "gene_name"])
    .to_csv("data/disease_gene/disease_associates_gene/disease_gene_summary_stats_dep_path.tsv", sep="\t", index=False)
)


# # Load the Newly Transformed Data

# In[11]:


candidate_df = (
    pd.read_table("data/disease_gene/disease_associates_gene/disease_gene_summary_stats_dep_path.tsv")
    .drop(['first_entity', 'first_entity_location',
       'second_entity', 'second_entity_location', 'first_entity_string',
       'second_entity_string', 'first_entity_db_id', 'second_entity_db_id',
       'first_entity_type', 'second_entity_type', 'dependency_path',
       'sentence_string', 'sentence_num', 'path', 'pubmed_id', 'Te', 'Te.ind'], axis=1)
)
candidate_df.head(2)


# In[12]:


prior_df = pd.read_csv("data/observation-prior.csv")
prior_df["logit_prior_perm"] = logit(prior_df["prior_perm"])
prior_df.head(2)


# # Set up the Training and Testing Set

# In[13]:


total_df = (
    pd.read_csv("data/disease-gene-pairs-association.csv.xz", compression='xz')
    .rename(index=str,columns={'entrez_gene_id': 'gene_id', 'doid_id': 'disease_id'})
)
total_df.head()


# In[14]:


# Gather the summary stats for each candidate
training_set = pd.merge(candidate_df, total_df.query("has_sentence==1&split==0")[["disease_id", "gene_id", "hetionet"]], 
                        how='inner',on=["disease_id", "gene_id"])
dev_set = pd.merge(candidate_df, total_df.query("has_sentence==1&split==1")[["disease_id", "gene_id", "hetionet"]],
                   how='inner', on=["disease_id", "gene_id"])
test_set = pd.merge(candidate_df, total_df.query("has_sentence==1&split==2")[["disease_id", "gene_id", "hetionet"]], 
                    how='inner', on=["disease_id", "gene_id"])


# Drop the values that aren't found in pubmed. 
training_set = training_set.drop("hetnet_labels", axis=1)
dev_set = dev_set.drop("hetnet_labels", axis=1)
test_set = test_set.drop("hetnet_labels", axis=1)

# Add the prior prob to the different sets 
training_set = pd.merge(training_set, prior_df[["disease_id", "gene_id", "logit_prior_perm"]], 
                        on=['disease_id', 'gene_id'])
dev_set = pd.merge(dev_set, prior_df[["disease_id", "gene_id", "logit_prior_perm"]], 
                  on=['disease_id', 'gene_id'])
test_set = pd.merge(test_set, prior_df[["disease_id", "gene_id", "logit_prior_perm"]], 
                   on=['disease_id', 'gene_id'])


# In[15]:


non_features = [
    "hetionet", "disease_id", "gene_id", 
    "gene_name", "disease_name",
    "pubmed_id"
]

X = training_set[[col for col in training_set.columns if col not in non_features]]
Y = training_set["hetionet"]

X_dev = dev_set[[col for col in dev_set.columns if col not in non_features]]
Y_dev = dev_set["hetionet"]

X_test = test_set[[col for col in test_set.columns if col not in non_features]]
Y_test = test_set["hetionet"]


# In[16]:


print(Y.value_counts())
print()
print(Y_dev.value_counts())
print()
print(Y_test.value_counts())
print()


# In[17]:


train_Y = Y.append(Y_dev)
train_X = X.append(X_dev)


# # Train the Machine Learning Algorithms

# Here we use gridsearch to optimize both models using 10 fold cross validation. After exhausting the list of parameters, the best model is chosen and analyzed in the next chunk. 

# In[18]:


n_iter = 100
final_models = []

lr = LogisticRegression()
lr_grid = {'C':np.linspace(1, 100, num=100)}

lr_normalizer_no_russ = StandardScaler()
lr_normalizer = StandardScaler()


# In[19]:


get_ipython().run_cell_magic('time', '', "\nruss_data = ['U', 'U.ind', 'Ud', 'Ud.ind', 'D', 'D.ind', 'J',\n       'J.ind', 'Y', 'Y.ind', 'G', 'G.ind', 'Md', 'Md.ind', 'X', 'X.ind', 'L',\n       'L.ind']\n\ntransformed_X_no_russ = lr_normalizer_no_russ.fit_transform(train_X[[col for col in train_X.columns if col not in russ_data]])\ntransformed_X = lr_normalizer.fit_transform(train_X)")


# In[20]:


get_ipython().run_cell_magic('time', '', "\nfinal_model = GridSearchCV(lr, lr_grid, cv=10, n_jobs=3, scoring='roc_auc', return_train_score=True)\nfinal_model.fit(transformed_X_no_russ, train_Y)\nfinal_models.append(final_model)")


# In[21]:


get_ipython().run_cell_magic('time', '', "\nfinal_model = GridSearchCV(lr, lr_grid, cv=10, n_jobs=3, scoring='roc_auc', return_train_score=True)\nfinal_model.fit(transformed_X, train_Y)\nfinal_models.append(final_model)")


# ## Parameter Optimization

# In[22]:


no_russ_result = pd.DataFrame(final_models[0].cv_results_)
russ_result = pd.DataFrame(final_models[1].cv_results_)


# In[23]:


# No LSTM
plt.plot(no_russ_result['param_C'], no_russ_result['mean_test_score'])


# In[24]:


# LSTM
plt.plot(russ_result['param_C'], russ_result['mean_test_score'])


# ## LR Weights

# In[25]:


list(zip(final_models[0].best_estimator_.coef_[0], [col for col in training_set.columns if col not in russ_data+non_features]))


# In[26]:


list(zip(final_models[1].best_estimator_.coef_[0], [col for col in training_set.columns if col not in non_features]))


# # Bar Plot AUROCS

# In[27]:


plt.figure(figsize=(10,5))
feature_rocs = []
for feature in X.columns:
    fpr, tpr, _ = roc_curve(train_Y, train_X[feature])
    feature_auc = auc(fpr, tpr)
    feature_rocs.append((feature, feature_auc))

feature_roc_df = pd.DataFrame(feature_rocs, columns=["Feature", "AUROC"])
ax = sns.barplot(x="AUROC", y="Feature", data=feature_roc_df, color='green')#palette=sns.color("Blue"))
plt.title("Training AUROC")
plt.xlim([0.4,1])


# In[28]:


plt.figure(figsize=(10,5))
feature_rocs = []
for feature in X.columns:
    fpr, tpr, _ = roc_curve(Y_test, X_test[feature])
    feature_auc = auc(fpr, tpr)
    feature_rocs.append((feature, feature_auc))

feature_roc_df = pd.DataFrame(feature_rocs, columns=["Feature", "AUROC"])
ax = sns.barplot(x="AUROC", y="Feature", data=feature_roc_df, color='green')
plt.title("Testing AUROC")
plt.xlim([0.4,1])


# # ROC CURVES

# In[ ]:


plt.figure(figsize=(12,8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

for feature in train_X:
    # Plot the p_values log transformed
    fpr, tpr, thresholds= roc_curve(train_Y, train_X[feature])
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="{} (area = {:0.2f})".format(feature, model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train ROC')
plt.legend(loc="lower right")


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

for feature in X:
    # Plot the p_values log transformed
    fpr, tpr, thresholds= roc_curve(Y_test, X_test[feature])
    model_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label="{} (area = {:0.2f})".format(feature, model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC')
plt.legend(loc="lower right")


# # Corerlation Matrix

# In[29]:


feature_corr_mat = train_X.corr()
sns.heatmap(feature_corr_mat, cmap="RdBu", center=0)


# # ML Performance

# In[30]:


transformed_tempX_test = lr_normalizer_no_russ.transform(X_test[[col for col in X.columns if col not in russ_data+non_features]])
transformed_X_test = lr_normalizer.transform(X_test)


# In[31]:


colors = ["green","red"]
labels = ["LR_NO_RUSS","LR_RUSS"]


# In[32]:


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

# Plot the p_values log transformed
fpr, tpr, thresholds= roc_curve(train_Y, train_X["logit_prior_perm"])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='cyan', label="{} (area = {:0.2f})".format("prior", model_auc))

fpr, tpr, thresholds= roc_curve(train_Y, final_models[0].predict_proba(transformed_X_no_russ)[:,1])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=colors[0], label="{} (area = {:0.2f})".format(labels[0], model_auc))

fpr, tpr, thresholds= roc_curve(train_Y, final_models[1].predict_proba(transformed_X)[:,1])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=colors[1], label="{} (area = {:0.2f})".format(labels[1], model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train ROC')
plt.legend(loc="lower right")


# In[33]:


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

# Plot the p_values log transformed
fpr, tpr, thresholds= roc_curve(Y_test, X_test["logit_prior_perm"])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='cyan', label="{} (area = {:0.2f})".format("prior", model_auc))

fpr, tpr, thresholds= roc_curve(Y_test, final_models[0].predict_proba(transformed_tempX_test)[:,1])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=colors[0], label="{} (area = {:0.2f})".format(labels[0], model_auc))

fpr, tpr, thresholds= roc_curve(Y_test, final_models[1].predict_proba(transformed_X_test)[:,1])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=colors[1], label="{} (area = {:0.2f})".format(labels[1], model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC')
plt.legend(loc="lower right")


# In[34]:


plt.figure()

# Plot the p_values log transformed
precision, recall, _= precision_recall_curve(Y_test, X_test["logit_prior_perm"])
model_precision = average_precision_score(Y_test, X_test["logit_prior_perm"])
plt.plot(recall, precision, color='cyan', label="{} (area = {:0.2f})".format("prior", model_precision))

precision, recall, _ = precision_recall_curve(Y_test, final_models[0].predict_proba(transformed_tempX_test)[:,1])
model_precision = average_precision_score(Y_test, final_models[0].predict_proba(transformed_tempX_test)[:,1])
plt.plot(recall, precision, color=colors[0], label="{} curve (area = {:0.2f})".format(labels[0], model_precision))
  
precision, recall, _ = precision_recall_curve(Y_test, final_models[1].predict_proba(transformed_X_test)[:,1])
model_precision = average_precision_score(Y_test, final_models[1].predict_proba(transformed_X_test)[:,1])
plt.plot(recall, precision, color=colors[1], label="{} curve (area = {:0.2f})".format(labels[1], model_precision))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Test Precision-Recall Curve')
plt.xlim([0, 1.01])
plt.ylim([0, 1.05])
plt.legend(loc="upper right")


# ## Save Final Result in DF

# In[ ]:


predictions = final_models[1].predict_proba(train_X.append(X_test))
predictions_df = training_set.append(dev_set).append(test_set)[[
    "disease_id","disease_name", 
    "gene_id", "gene_name", 'hetnet']]
predictions_df["predictions"] = predictions[:,1]


# In[ ]:


predictions_df.to_csv("data/vanilla_lstm/final_model_predictions.csv", index=False)

