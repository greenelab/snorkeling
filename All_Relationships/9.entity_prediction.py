
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
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[2]:


candidate_df = pd.read_csv("data/disease_gene_summary_stats.csv")
prior_df = pd.read_csv("data/observation-prior.csv")


# In[3]:


candidate_df.head(10)


# In[4]:


prior_df.head(10)


# # Set up the Training and Testing Set

# In[5]:


train_df = pd.read_csv("stratified_data/train_set.csv")
dev_df = pd.read_csv("stratified_data/dev_set.csv")
test_df = pd.read_csv("stratified_data/test_set.csv")


# In[6]:


# Gather the summary stats for each candidate
training_set = pd.merge(candidate_df, train_df.query("pubmed==1")[["disease_id", "gene_id", "hetnet"]], 
                        how='inner', on=["disease_id", "gene_id"])
dev_set = pd.merge(candidate_df, dev_df.query("pubmed==1")[["disease_id", "gene_id", "hetnet"]],
                   how='inner', on=["disease_id", "gene_id"])
test_set = pd.merge(candidate_df, test_df.query("pubmed==1")[["disease_id", "gene_id", "hetnet"]], 
                    how='inner', on=["disease_id", "gene_id"])


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


non_features = [
    "hetnet", "disease_id", "gene_id", 
    "gene_name", "disease_name",
    "pubmed", "lstm_marginal_0_quantile", 
    "lstm_marginal_20_quantile","lstm_marginal_40_quantile",
    "lstm_marginal_60_quantile", "lstm_marginal_80_quantile"
]

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

no_lstm_normalizer = StandardScaler()
lstm_normalizer = StandardScaler()


# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Train on data without LSTM input\nlstm_features = [\n    "lstm_avg_marginal"\n]\n\ntempX = X[[col for col in X.columns if col not in lstm_features]]\ntempX = tempX.append(X_dev[[col for col in X_dev.columns if col not in lstm_features]])\n\ntransformed_tempX = no_lstm_normalizer.fit_transform(tempX)\ntransformed_X = lstm_normalizer.fit_transform(X.append(X_dev))')


# In[10]:


get_ipython().run_cell_magic('time', '', "\nfinal_model = GridSearchCV(lr, lr_grid, cv=10, n_jobs=3, scoring='roc_auc', return_train_score=True)\nfinal_model.fit(transformed_tempX, Y.append(Y_dev))\nfinal_models.append(final_model)")


# In[11]:


get_ipython().run_cell_magic('time', '', "\n# Train on data with LSTM input\nfinal_model = GridSearchCV(lr, lr_grid, cv=10, n_jobs=3, scoring='roc_auc', return_train_score=True)\nfinal_model.fit(transformed_X, Y.append(Y_dev))\nfinal_models.append(final_model)")


# ## Parameter Optimization

# In[12]:


no_lstm_result = pd.DataFrame(final_models[0].cv_results_)
lstm_result = pd.DataFrame(final_models[1].cv_results_)


# In[13]:


# No LSTM
plt.plot(no_lstm_result['param_C'], no_lstm_result['mean_test_score'])


# In[14]:


# LSTM
plt.plot(lstm_result['param_C'], lstm_result['mean_test_score'])


# ## LR Weights

# In[15]:


list(zip(final_models[0].best_estimator_.coef_[0], [col for col in training_set.columns if col not in lstm_features+non_features]))


# In[16]:


list(zip(final_models[1].best_estimator_.coef_[0], [col for col in training_set.columns if col not in non_features]))


# # AUROCS

# In[17]:


feature_rocs = []
train_Y = Y.append(Y_dev)
train_X = X.append(X_dev)
for feature in X.columns:
    fpr, tpr, _ = roc_curve(train_Y, train_X[feature])
    feature_auc = auc(fpr, tpr)
    feature_rocs.append((feature, feature_auc))

feature_roc_df = pd.DataFrame(feature_rocs, columns=["Feature", "AUROC"])
ax = sns.barplot(x="AUROC", y="Feature", data=feature_roc_df)
plt.xlim([0.5,1])


# # Corerlation Matrix

# In[18]:


feature_corr_mat = train_X.corr()
sns.heatmap(feature_corr_mat, cmap="RdBu")


# # ML Performance

# In[19]:


transformed_tempX_test = no_lstm_normalizer.transform(X_test[[col for col in X.columns if col not in lstm_features]])
transformed_X_test = lstm_normalizer.transform(X_test)


# In[20]:


colors = ["green","red"]
labels = ["LR_NO_LSTM","LR_LSTM"]


# In[21]:


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")

# Plot the p_values log transformed
fpr, tpr, thresholds= roc_curve(Y_test, X_test["prior_perm"])
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
plt.title('ROC')
plt.legend(loc="lower right")


# In[22]:


plt.figure()

# Plot the p_values log transformed
precision, recall, _= precision_recall_curve(Y_test, X_test["prior_perm"])
model_precision = average_precision_score(Y_test, X_test["prior_perm"])
plt.plot(recall, precision, color='cyan', label="{} (area = {:0.2f})".format("prior", model_precision))

precision, recall, _ = precision_recall_curve(Y_test, final_models[0].predict_proba(transformed_tempX_test)[:,1])
model_precision = average_precision_score(Y_test, final_models[0].predict_proba(transformed_tempX_test)[:,1])
plt.plot(recall, precision, color=colors[0], label="{} curve (area = {:0.2f})".format(labels[0], model_precision))
  
precision, recall, _ = precision_recall_curve(Y_test, final_models[1].predict_proba(transformed_X_test)[:,1])
model_precision = average_precision_score(Y_test, final_models[1].predict_proba(transformed_X_test)[:,1])
plt.plot(recall, precision, color=colors[1], label="{} curve (area = {:0.2f})".format(labels[1], model_precision))

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision-Recall Curve')
plt.xlim([0, 1.01])
plt.ylim([0, 1.05])
plt.legend(loc="upper right")


# ## Save Final Result in DF

# In[ ]:


predictions = final_models[1].predict_proba(X.append(X_dev).append(X_test))
predictions_df = training_set.append(dev_set).append(test_set)[[
    "disease_id","disease_name", 
    "gene_id", "gene_name", 'hetnet']]
predictions_df["predictions"] = predictions[:,1]


# In[ ]:


predictions_df.to_csv("data/vanilla_lstm/final_model_predictions.csv", index=False)

