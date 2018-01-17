
# coding: utf-8

# # Train/Test the True Relationship Model

# This notebook is design to predict DG relationships on the entity level. Here we are taking the input from the Bi-LSTM model, prior probability notebook and the summary statistics notebook and combinging it into a single dataset. From there we train a Ridge LR model and an elastic net LR model to make the final prediction.

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
from scipy.stats import fisher_exact
import scipy
from sqlalchemy import and_
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sns


# In[2]:


candidate_df = pd.read_csv("disease_gene_summary_stats.csv")
prior_df = pd.read_csv("observation-prior.csv")


# In[3]:


candidate_df.head(10)


# In[4]:


prior_df.head(10)


# # Set up the Training and Testing Set

# In[5]:


train_df = pd.read_csv("stratified_data/training_set.csv")
dev_df = pd.read_csv("stratified_data/dev_set.csv")
test_df = pd.read_csv("stratified_data/test_set.csv")


# In[6]:


# Gather the summary stats for each candidate
training_set = pd.merge(candidate_df, train_df, how='right', on=["disease_id", "gene_id"])
dev_set = pd.merge(candidate_df, dev_df, how='right', on=["disease_id", "gene_id"])
test_set = pd.merge(candidate_df, test_df, how='right', on=["disease_id", "gene_id"])


# Drop the values that aren't found in pubmed. 
training_set = training_set.drop("hetnet_labels", axis=1)
dev_set = dev_set.drop("hetnet_labels", axis=1)
test_set = test_set.drop("hetnet_labels", axis=1)

training_set = training_set.dropna()
dev_set = dev_set.dropna()
test_set = test_set.dropna()

# Add the prior prob to the different sets 
training_set = pd.merge(training_set, prior_df[["disease_id", "gene_id", "prior_perm"]])
dev_set = pd.merge(dev_set, prior_df[["disease_id", "gene_id", "prior_perm"]])
test_set = pd.merge(test_set, prior_df[["disease_id", "gene_id", "prior_perm"]])


# In[7]:


non_features = ["hetnet","final_model_pred", "disease_id", "gene_id", "gene_name", "disease_name", "pubmed"]

X = training_set[[col for col in training_set.columns if col not in non_features]]
Y = training_set["hetnet"]

X_dev = dev_set[[col for col in dev_set.columns if col not in non_features]]
Y_dev = dev_set["hetnet"]

X_test = test_set[[col for col in test_set.columns if col not in non_features]]
Y_test = test_set["hetnet"]


# # Train the Machine Learning Algorithms

# Here we use gridsearch to optimize both models using 10 fold cross validation. After exhausting the list of parameters, the best model is chosen and analyzed in the next chunk. 

# In[8]:


n_iter = 100
final_models = []

lr = LogisticRegression()
lr_grid = {'C':np.linspace(1, 100, num=100)}


# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Train on data without LSTM input\nlstm_features = [\n    "avg_marginal", "quantile_zero", \n    "quantile_twenty","quantile_forty",\n    "quantile_sixty", "quantile_eighty", \n    "lower_ci"\n]\n\n\ntempX = X[[col for col in X.columns if col not in lstm_features]]\ntempX = tempX.append(X_dev[[col for col in X_dev.columns if col not in lstm_features]])\n\nfinal_model = GridSearchCV(lr, lr_grid, cv=10, n_jobs=3, scoring=\'roc_auc\')\nfinal_model.fit(tempX, Y.append(Y_dev))\nfinal_models.append(final_model)')


# In[10]:


get_ipython().run_cell_magic('time', '', '\n# Train on data with LSTM input\nfinal_model = GridSearchCV(lr, lr_grid, cv=10, n_jobs=3)\nfinal_model.fit(X.append(X_dev), Y.append(Y_dev))\nfinal_models.append(final_model)')


# ## Parameter Optimization

# In[11]:


no_lstm_result = pd.DataFrame(final_models[0].cv_results_)
lstm_result = pd.DataFrame(final_models[1].cv_results_)


# In[12]:


# No LSTM
plt.plot(no_lstm_result['param_C'], no_lstm_result['mean_test_score'])


# In[13]:


# LSTM
plt.plot(lstm_result['param_C'], lstm_result['mean_test_score'])


# ## LR Weights

# In[14]:


zip(final_models[0].best_estimator_.coef_[0], [col for col in training_set.columns if col not in lstm_features+non_features])


# In[15]:


zip(final_models[1].best_estimator_.coef_[0], [col for col in training_set.columns if col not in non_features])


# # ML Performance

# In[16]:


colors = ["green","red"]
labels = ["LR_NO_LSTM","LR_LSTM"]


# In[17]:


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

# Plot the p_values log transformed
fpr, tpr, thresholds= roc_curve(Y_test, X_test["prior_perm"])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='cyan', label="{} (area = {:0.2f})".format("prior", model_auc))

fpr, tpr, thresholds= roc_curve(Y_test, final_models[0].predict_proba(X_test[[col for col in X.columns if col not in lstm_features]])[:,1])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=colors[0], label="{} (area = {:0.2f})".format(labels[0], model_auc))

fpr, tpr, thresholds= roc_curve(Y_test, final_models[1].predict_proba(X_test)[:,1])
model_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color=colors[1], label="{} (area = {:0.2f})".format(labels[1], model_auc))

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")


# In[18]:


plt.figure()

# Plot the p_values log transformed
precision, recall, _= precision_recall_curve(Y_test, X_test["prior_perm"])
model_precision = average_precision_score(Y_test, X_test["prior_perm"])
plt.plot(recall, precision, color='cyan', label="{} (area = {:0.2f})".format("prior", model_precision))

precision, recall, _ = precision_recall_curve(Y_test, final_models[0].predict_proba(X_test[[col for col in X.columns if col not in lstm_features]])[:,1])
model_precision = average_precision_score(Y_test, final_models[0].predict_proba(X_test[[col for col in X.columns if col not in lstm_features]])[:,1])
plt.plot(recall, precision, color=colors[0], label="{} curve (area = {:0.2f})".format(labels[0], model_precision))
  
precision, recall, _ = precision_recall_curve(Y_test, final_models[1].predict_proba(X_test)[:,1])
model_precision = average_precision_score(Y_test, final_models[1].predict_proba(X_test)[:,1])
plt.plot(recall, precision, color=colors[1], label="{} curve (area = {:0.2f})".format(labels[1], model_precision))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.xlim([0, 1.01])
plt.ylim([0, 1.05])
plt.legend(loc="upper right")


# ## Save Final Result in DF

# In[19]:


predictions = final_models[1].predict_proba(X_test[[col for col in training_set.columns if col not in ["hetnet","final_model_pred", "disease_id", "gene_id", "gene_name", "disease_name", "pubmed"]]])
predictions_df = pd.DataFrame([], columns=["predictions"])
predictions_df["predictions"] = predictions[:,1]


# In[20]:


predictions_df.to_csv("final_model_predictions.csv", index=False)

