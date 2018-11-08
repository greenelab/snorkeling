
# coding: utf-8

# # Generative Model Benchmarking

# The goal here is to use the [data programing paradigm](https://arxiv.org/abs/1605.07723) to probabilistically label our training dataset for the disease associates gene relationship. The label functions have already been generated and now it is time to train the generative model. This model captures important features such as agreements and disagreements between label functions, by estimating the probability of label functions emitting a combination of labels given the class. $P(\lambda_{i} = j \mid Y=y)$. More information can be found in this [technical report](https://arxiv.org/pdf/1810.02840.pdf) or in this [paper](https://ajratner.github.io/assets/papers/deem-metal-prototype.pdf). The testable hypothesis here is: **Incorporating multiple weak sources improves performance compared to the normal distant supervision approach, which uses a single resource for labels**.

# # Experimental Design:
# 
# Compares three different models. The first model uses four databases (DisGeNET, Diseases, DOAF and GWAS) as the distant supervision approach. The second model uses the above databases with user defined rules such as (regular expressions, trigger word identification and sentence contextual rules). The last model uses the above sources of information in conjunction with biclustering data obtained from this [paper](https://www.ncbi.nlm.nih.gov/pubmed/29490008).

# ## Dataset
#     
# | Set type  | Size |
# |:---|:---|
# | Train |  50k  |
# | Dev |  210 (hand labeled) |

# ## Set up The Environment

# The few blocks below sets up our python environment to perform the experiment.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from itertools import product
import os
import pickle
import sys

sys.path.append(os.path.abspath('../../../modules'))

import matplotlib.pyplot as plt
import pandas as pd
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


from snorkel.annotations import LabelAnnotator
from snorkel.learning.structure import DependencySelector
from snorkel.models import candidate_subclass

from metal.analysis import confusion_matrix
from metal.label_model import LabelModel
from metal.utils import convert_labels
from metal.contrib.visualization.analysis import(
    plot_predictions_histogram, 
)

from utils.label_functions import DG_LFS

from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.label_matrix_helper import (
    get_auc_significant_stats, 
    get_overlap_matrix, 
    get_conflict_matrix, 
    label_candidates
)
from utils.notebook_utils.train_model_helper import train_generative_model
from utils.notebook_utils.plot_helper import (
    plot_label_matrix_heatmap, 
    plot_curve, 
    plot_generative_model_weights, 
)


# In[4]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[5]:


quick_load = True


# ## Load the data for Generative Model Experiments

# In[6]:


spreadsheet_names = {
    'train': '../../sentence_labels_train.xlsx',
    'dev': '../../sentence_labels_dev.xlsx',
    'test': '../../sentence_labels_test.xlsx'
}


# In[7]:


candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key])
    for key in spreadsheet_names
}

for key in candidate_dfs:
    print("Size of {} set: {}".format(key, candidate_dfs[key].shape[0]))


# In[8]:


label_functions = (
    list(DG_LFS["DaG"].values())
) 

if quick_load:
    label_matricies = pickle.load(open("label_matricies.pkl", "rb"))
else:
    #labeler = LabelAnnotator(lfs=label_functions)
    label_matricies = {
        key:label_candidates(
            session,
            candidate_dfs[key]['candidate_id'],
            label_functions,
            num_threads=10, 
            batch_size=candidate_dfs[key]['candidate_id'].shape[0]
        )
        for key in candidate_dfs
    }


# In[9]:


lf_names = list(DG_LFS["DaG"].keys())


# ## Visualize Label Functions

# Before training the generative model, here are some visualizations for the given label functions. These visualizations are helpful in determining the efficacy of each label functions as well as observing the overlaps and conflicts between each function.

# In[10]:


plt.rcParams.update({'font.size': 10})
plot_label_matrix_heatmap(label_matricies['train'].T, 
                          yaxis_tick_labels=lf_names, 
                          figsize=(10,8), font_size=10)


# Looking at the heatmap above, this is a decent distribution of labels. Some of the label functions are covering a lot of data points (distant supervision ones) and some are very sparse in their output.

# In[11]:


plot_label_matrix_heatmap(get_overlap_matrix(label_matricies['train'], normalize=True), 
                          yaxis_tick_labels=lf_names, xaxis_tick_labels=lf_names,
                          figsize=(10,8), colorbar=False, plot_title="Overlap Matrix")


# The overlap matrix above shows how two label functions overlap with each other. The brighter the color the more overlaps a label function has with another label function.

# In[12]:


plot_label_matrix_heatmap(get_conflict_matrix(label_matricies['train'], normalize=True), 
                          yaxis_tick_labels=lf_names, xaxis_tick_labels=lf_names,
                          figsize=(10,8), colorbar=False, plot_title="Conflict Matrix")


# The conflict matrix above shows how often label functions conflict with each other. The brighter the color the more conflict a label function has with another function. Ignoring the diagonals, there isn't many conflicts between functions except for the LF_DG_NO_CONCLUSION and LF_DG_ALLOWED_DISTANCE.

# # Train the Generative Model

# After visualizing the label functions and their associated properties, now it is time to work on the generative model. As with common machine learning pipelines, the first step is to find the best hyperparameters for this model. Using the grid search algorithm, the follow parameters were optimized: amount of burnin, strength of regularization, number of epochs to run the model.

# ## Set the hyperparameter grid search

# In[13]:


regularization_grid = pd.np.round(pd.np.linspace(0.1, 6, num=25), 3)


# ## What are the best hyperparameters for the conditionally independent model?

# In[14]:


L = convert_labels(label_matricies['train'].toarray(), 'plusminus', 'categorical')
L_dev = convert_labels(label_matricies['dev'].toarray(), 'plusminus', 'categorical')
L_test = convert_labels(label_matricies['test'].toarray(), 'plusminus', 'categorical')

validation_data = list(zip([L[:,:7], L[:, :24], L], [L_dev[:,:7], L_dev[:, :24], L_dev]))
test_data = list(zip([L[:,:7], L[:, :24], L], [L_test[:,:7], L_test[:, :24], L_test]))
model_labels = ["Distant Supervision (DS)", "DS+User Defined Rules", "All"]


# In[15]:


model_grid_search = {}
for model_data, model_label in zip(validation_data, model_labels):
    
    label_model = LabelModel(k=2, seed=100)
    grid_results = {}
    for param in regularization_grid:
        label_model.train_model(model_data[0], n_epochs=1000, verbose=False, lr=0.01, l2=param)
        grid_results[str(param)] = label_model.predict_proba(model_data[1])[:,0]
        
    model_grid_search[model_label] = pd.DataFrame.from_dict(grid_results)


# In[16]:


model_grid_aucs = {}
for model in model_grid_search:
    model_grid_aucs[model] = plot_curve(model_grid_search[model], candidate_dfs['dev'].curated_dsh, 
                               figsize=(16,6), model_type='scatterplot', plot_title=model, metric="ROC", font_size=10)


# In[17]:


model_grid_auc_dfs = {}
for model in model_grid_aucs:
    model_grid_auc_dfs[model] = (
        get_auc_significant_stats(candidate_dfs['dev'], model_grid_aucs[model])
        .sort_values('auroc', ascending=False)
    )
    print(model)
    print(model_grid_auc_dfs[model].head(5))
    print()


# # Final Evaluation on Held out Hand Labeled Test Data

# In[18]:


dev_model_df = pd.DataFrame()
for best_model, model_data, model_label in zip([1.083, 2.067, 1.575], validation_data, model_labels):
    label_model = LabelModel(k=2, seed=100)
    label_model.train_model(model_data[0] , n_epochs=1000, verbose=False, lr=0.01, l2=best_model)
    dev_model_df[model_label] = label_model.predict_proba(model_data[1])[:,0]


# In[19]:


_ = plot_curve(
    dev_model_df, 
    candidate_dfs['dev'].curated_dsh,
    model_type='curve', figsize=(10,8), 
    plot_title="Disease Associates Gene AUROC on Dev Data", font_size=16
)


# In[20]:


_ = plot_curve(
    dev_model_df, 
    candidate_dfs['dev'].curated_dsh,
    model_type='curve', figsize=(12,7), 
    plot_title="Disease Associates Gene Dev PRC",
    metric='PR', font_size=16
)


# In[21]:


label_model = LabelModel(k=2, seed=100)
label_model.train_model(validation_data[1][0], n_epochs=1000, verbose=False, lr=0.01, l2=2.067)
dev_predictions = convert_labels(label_model.predict(validation_data[1][1]), 'categorical', 'onezero')
dev_marginals = label_model.predict_proba(validation_data[1][1])[:,0]


# In[22]:


plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(10,6))
plot_predictions_histogram(
    dev_predictions,
    candidate_dfs['dev'].curated_dsh.astype(int).values,
    title="Prediction Histogram for Dev Set"
)


# In[23]:


confusion_matrix(
    convert_labels(candidate_dfs['dev'].curated_dsh.values, 'onezero', 'categorical'),
    convert_labels(dev_predictions, 'onezero', 'categorical')
)

