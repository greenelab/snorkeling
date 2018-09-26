
# coding: utf-8

# # Generative Model Benchmarking

# The goal here is to use the [data programing paradigm](https://arxiv.org/abs/1605.07723) to probabilistically label our training dataset for the disease associates gene realtionship. The label functions have already been generated and now it is time to train the generative model. This model captures important features such as agreements and disagreements between label functions; furthermore, this model can capture the dependency structure between label functions (i.e. correlations between label functions). More information can be found in this [blog post](https://hazyresearch.github.io/snorkel/blog/structure_learning.html) or in this [paper](https://arxiv.org/abs/1703.00854). The underlying hypothesis here is: **Modeling dependency structure between label functions has better performance compared to the conditionally independent model.**

# ## Set up The Environment

# The few blocks below sets up our python environment to perform the experiment.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from itertools import product
import os
import sys

sys.path.append(os.path.abspath('../../../modules'))

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

from utils.label_functions import DG_LFS

from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.label_matrix_helper import *
from utils.notebook_utils.train_model_helper import train_generative_model
from utils.notebook_utils.plot_helper import plot_label_matrix_heatmap, plot_roc_curve, plot_generative_model_weights, plot_pr_curve


# In[4]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[5]:


quick_load = True


# ## Load the data for Generative Model Experiments

# In[6]:


spreadsheet_names = {
    'train': '../../sentence_labels_train.xlsx',
    'dev': '../../sentence_labels_train_dev.xlsx',
    'test': '../../sentence_labels_dev.xlsx'
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
    list(DG_LFS["DaG_DB"].values()) + 
    list(DG_LFS["DaG_TEXT"].values())
) 

if quick_load:
    labeler = LabelAnnotator(lfs=[])

    label_matricies = {
        key:labeler.load_matrix(session, cids_query=make_cids_query(session, candidate_dfs[key]))
        for key in candidate_dfs
    }

else:
    labeler = LabelAnnotator(lfs=label_functions)

    label_matricies = {
        key:label_candidates(
            labeler, 
            cids_query=make_cids_query(session, candidate_dfs[key]),
            label_functions=label_functions,
            apply_existing=(key!='train')
        )
        for key in candidate_dfs
    }


# In[9]:


lf_names = [
    label_matricies['test'].get_key(session, index).name 
    for index in range(label_matricies['test'].shape[1])
]


# ## Visualize Label Functions

# Before training the generative model, here are some visualizations for the given label functions. These visualizations are helpful in determining the efficacy of each label functions as well as observing the overlaps and conflicts between each function.

# In[10]:


plot_label_matrix_heatmap(label_matricies['train'].T, 
                          yaxis_tick_labels=lf_names, 
                          figsize=(10,8))


# Looking at the heatmap above, this is a decent distribution of labels. Some of the label functions are outputting a lot of labels (distant supervision ones) and some are very sparse in their output. Nevertheless, nothing shocking scream out here in terms of label function performance. 

# In[11]:


plot_label_matrix_heatmap(get_overlap_matrix(label_matricies['train'], normalize=True), 
                          yaxis_tick_labels=lf_names, xaxis_tick_labels=lf_names,
                          figsize=(10,8), colorbar=False, plot_title="Overlap Matrix")


# The overlap matrix above shows how two label functions overlap with each other. The brighter the color the more overlaps a label function has with another label function. Ignoring the diagonals, there isn't much overlap between functions as expected.

# In[12]:


plot_label_matrix_heatmap(get_conflict_matrix(label_matricies['train'], normalize=True), 
                          yaxis_tick_labels=lf_names, xaxis_tick_labels=lf_names,
                          figsize=(10,8), colorbar=False, plot_title="Conflict Matrix")


# The conflict matrix above shows how often label functions conflict with each other. The brighter the color the more conflict a label function has with another function. Ignoring the diagonals, there isn't many conflicts between functions except for the LF_DG_NO_CONCLUSION and LF_DG_ALLOWED_DISTANCE. Possible reasons for lack of conflicts could be lack of coverage a few functions have, which is shown in the cell below.

# In[13]:


label_matricies['train'].lf_stats(session)


# # Train the Generative Model

# After visualizing the label functions and their associated properties, now it is time to work on the generative model. AS with common machine learning pipelines, the first step is to find the best hyperparameters for this model. Using the grid search algorithm, the follow parameters were optimized: amount of burnin, strength of regularization, number of epochs to run the model.

# ## Set the hyperparameter grid search

# In[14]:


regularization_grid = np.round(pd.np.linspace(0.001, 0.5, num=20), 3)


# ## What are the best hyperparameters for the conditionally independent model?

# In[15]:


gen_ci_models = {
   "{}".format(str(parameter)):train_generative_model(
        label_matricies['train'],
        burn_in=100,
        epochs=100,
        reg_param=parameter,
        step_size=1/label_matricies['train'].shape[0]
    )
    for parameter in tqdm_notebook(regularization_grid)
}


# In[16]:


ci_marginal_df = pd.DataFrame(pd.np.array([
    gen_ci_models[model_name].marginals(label_matricies['dev'])
    for model_name in sorted(gen_ci_models.keys())
]).T, columns=sorted(gen_ci_models.keys()))
ci_marginal_df['candidate_id'] = candidate_dfs['dev'].candidate_id.values
ci_marginal_df.head(2)


# In[31]:


ci_aucs = plot_roc_curve(
    ci_marginal_df.drop("candidate_id", axis=1), 
    candidate_dfs['dev'].curated_dsh,
    model_type='scatterplot', xlim=[0,0.7], figsize=(12,8), 
    plot_title="Disease Associates Gene CI AUROC"
)


# In[18]:


ci_auc_stats_df = get_auc_significant_stats(candidate_dfs['dev'], ci_aucs).sort_values('auroc', ascending=False)
ci_auc_stats_df


# From this data frame, the best performing model had the following parameters: 50-burnin, 50-epochs, 0.2-regularization. By looking at the top five models, the regularization parameter stays at 0.2. The amount of epochs and burnin varies, but the regularization parameter is important to note.

# In[30]:


plot_pr_curve(
    ci_marginal_df.drop("candidate_id", axis=1), 
    candidate_dfs['dev'].curated_dsh,
    model_type='scatterplot', xlim=[0, 1], figsize=(12,8), 
    plot_title="Disease Associates Gene CI AUPRC"
)


# In[20]:


best_model_ci = ci_auc_stats_df.iloc[0].name
test_ci_marginal_df = pd.DataFrame(pd.np.array([
    gen_ci_models[ci_auc_stats_df.iloc[0].name].marginals(label_matricies['test'])
]).T, columns=[best_model_ci])
test_ci_marginal_df['candidate_id'] = candidate_dfs['test'].candidate_id.values
test_ci_marginal_df.head(2)


# ## Does modeling dependencies aid in performance?

# In[21]:


from snorkel.learning.structure import DependencySelector
gen_da_models = {
    "{}".format(parameter):train_generative_model(
        label_matricies['train'],
        burn_in=100,
        epochs=100,
        reg_param=parameter,
        step_size=1/label_matricies['train'].shape[0],
        deps=DependencySelector().select(label_matricies['train']),
        lf_propensity=True
    )
    for parameter in tqdm_notebook(regularization_grid)
}


# In[22]:


da_marginal_df = pd.DataFrame(pd.np.array([
    gen_da_models[model_name].marginals(label_matricies['dev'])
    for model_name in sorted(gen_da_models.keys())
]).T, columns=sorted(gen_da_models.keys()))
da_marginal_df['candidate_id'] = candidate_dfs['dev'].candidate_id.values
da_marginal_df.head(2)


# In[29]:


da_aucs = plot_roc_curve(
    da_marginal_df.drop("candidate_id", axis=1), 
    candidate_dfs['dev'].curated_dsh,
    model_type='scatterplot', xlim=[0,1], figsize=(12,8),
    plot_title="Disease Associates Gene DA AUROC"
)


# In[24]:


da_auc_stats_df = get_auc_significant_stats(candidate_dfs['dev'], da_aucs).sort_values('auroc', ascending=False)
da_auc_stats_df


# From this data frame, the best performing model had the following parameters: 100-burnin, 100-epochs, 0.2-regularization. By looking at the top nine models, the regularization parameter stays at 0.2. The pattern of regularization is the same with the conditionally independent model. This means using 0.2 is a good choice for regularization. The amount of burnin and epochs can vary.

# In[32]:


plot_pr_curve(
    da_marginal_df.drop("candidate_id", axis=1), 
    candidate_dfs['dev'].curated_dsh,
    model_type='scatterplot', xlim=[0, 1], figsize=(12,8),
    plot_title="Disease Associates Gene DA AUPRC"
)


# In[26]:


best_model_da = da_auc_stats_df.iloc[0].name
test_da_marginal_df = pd.DataFrame(pd.np.array([
    gen_da_models[da_auc_stats_df.iloc[0].name].marginals(label_matricies['test'])
]).T, columns=[best_model_da])
test_da_marginal_df['candidate_id'] = candidate_dfs['test'].candidate_id.values
test_da_marginal_df.head(2)


# In[27]:


combined_df = pd.DataFrame(np.array([
    test_da_marginal_df.drop("candidate_id", axis=1)[best_model_da].values,
    test_ci_marginal_df.drop("candidate_id", axis=1)[best_model_ci].values
]).T, columns=["DA", "CI"])


# In[28]:


_ = plot_roc_curve(
    combined_df, 
    candidate_dfs['test'].curated_dsh,
    model_type='curve', xlim=[0,0.7], figsize=(10,8), 
    plot_title="Disease Associates Gene Test AUROC"
)


# Printed above are the best performing models from the conditinally independent model and the dependency aware model. These reults support the hypothesis that modeling depenency structure improves performance compared to the conditionally indepent assumption. Now that the best parameters are found the next step is to begin training the discriminator model to make the actual classification of sentneces.
