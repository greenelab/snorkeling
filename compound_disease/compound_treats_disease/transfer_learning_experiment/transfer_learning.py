
# coding: utf-8

# # Transfer Learning for Compound Treats Disease Relationship

# Based on previous experiments, the next step is to answer the question: Can label functions transfer between realtionship types? This notebook is designed to answer this question by testing the hypothesis: **Compound Binds Gene (CbG) label functions transfer better than Disease Associates Gene (DaG) functions when classifiying Compound Treats Disease sentences.** To test this hypothesis, we plan to train multiple generative models and measure each models performance in the form of Receiver Operative Cruves (ROC) and Precision-Recall Curves (PRCs) 

# ## Set up the Environment

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import os
import sys

sys.path.append(os.path.abspath('../../../modules'))
sys.path.append(os.path.abspath('data/label_functions/'))

import pandas as pd
from tqdm import tqdm_notebook


# In[ ]:


#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()


# In[ ]:


from snorkel.annotations import LabelAnnotator
from snorkel.models import candidate_subclass

from metal.label_model import LabelModel

from compound_disease_lf import CD_LFS
from disease_gene_lf import DG_LFS
from compound_gene_lf import CG_LFS
from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.label_matrix_helper import (
    get_columns, label_candidates, get_auc_significant_stats
)
from utils.notebook_utils.plot_helper import plot_label_matrix_heatmap, plot_roc_curve, plot_generative_model_weights


# In[ ]:


CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])


# In[ ]:


quick_load = True


# ## Load the Label Matrix

# Before we begin training the generative model, we need to load label matricies for each training/testing set. This process involves extracting annotations from the postgress database shared on this local machine. 

# In[ ]:


spreadsheet_names = {
    'train': 'data/sentence_labels_train.xlsx',
    'dev': 'data/sentence_labels_train_dev.xlsx',
    'test': 'data/sentence_labels_dev.xlsx'
}


# In[ ]:


candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key])
    for key in spreadsheet_names
}

for key in candidate_dfs:
    print("Size of {} set: {}".format(key, candidate_dfs[key].shape[0]))


# In[ ]:


label_functions = (
    list(CD_LFS['CtD_DB'].values()) + 
    list(CD_LFS['CtD_TEXT'].values()) + 
    list(CD_LFS['CD_BICLUSTER'].values()) + 
    list(CG_LFS["CbG_TEXT"].values())+
    list(DG_LFS["DaG_TEXT"].values())
) 

if quick_load:
    label_matricies = pickle.load(open("data/label_matricies.pkl", "rb"))

else:
    label_matricies = {
        key:label_candidates(
            session,
            candidates_dfs[key]['candidate_id'],
            label_functions,
            num_threads=10,
            batch_size=candidates_dfs[key]['candidate_id'].shape[0]
        )
        for key in candidate_dfs
    }
    
    pickle.dump(label_matricies, open("data/label_matricies.pkl", "wb"))


# In[ ]:


lf_names = [
    label_matricies['test'].get_key(session, index).name 
    for index in range(label_matricies['test'].shape[1])
]


# ## Observe Label Function Properties

# Before training the generative model, it is a good idea to observe label functions and their output. In the cells below we take a look at generated heatmaps and observe the underlying structure.

# In[ ]:


plot_label_matrix_heatmap(label_matricies['train'].T, 
                          yaxis_tick_labels=lf_names, 
                          figsize=(10,13))


# # Train Generative Model

# In[ ]:


cd_db = get_columns(session, label_matricies['train'], CD_LFS, "CtD_DB")
cd_text = get_columns(session, label_matricies['train'], CD_LFS, "CtD_TEXT")
cd_bicluster = get_columns(session, label_matricies['train'], CD_LFS, "CD_BICLUSTER")
cg_text = get_columns(session, label_matricies['train'], CG_LFS, "CbG_TEXT")
dg_text = get_columns(session, label_matricies['train'], DG_LFS, "DaG_TEXT")


# In[ ]:


# Dictionary specifying the different models 
# for this analysis
model_dict = {
    "CtD_DB": cd_db,
    "CtD_TEXT": cd_text,
    "CD_BICLUSTER": cd_bicluster,
    "CbG_TEXT": cg_text,
    "DaG_TEXT": dg_text,
    "CtD_DB_TEXT": cd_db+cd_text,
    "CtD_ALL": cd_db+cd_text+cd_bicluster,
    "All_the_jawns": cd_db+cd_text+cd_bicluster+cg_text+dg_text
}   


# In[ ]:


L = label_matricies['train'].toarray()
L[L < 0] = 2
L_dev = label_matricies['dev'].toarray()
L_dev[L_dev < 0] = 2
L_test = label_matricies['test'].toarray()
L_test[L_test < 0] = 2

label_model = LabelModel(k=2, seed=100)


# In[ ]:


reg_param_grid = pd.np.round(pd.np.linspace(1e-1, 1, num=30), 3)
grid_results = defaultdict(dict)
for model in tqdm_notebook(model_dict):
    for reg_param in reg_param_grid:
        label_model.train(L[:, model_dict[model]], n_epochs=1000, verbose=False, lr=0.01, l2=reg_param)
        grid_results[model][str(reg_param)] = label_model.predict_proba(L_dev[:, model_dict[model]])[:,0]


# In[ ]:


for model in grid_results:
    model_aucs = plot_roc_curve(pd.DataFrame.from_dict(grid_results[model]), candidate_dfs['dev'].curated_dsh, 
                               figsize=(16,6), model_type='scatterplot', plot_title=model)


# In[ ]:


best_params = {
    "CtD_DB":0.1,
    "CtD_TEXT": 0.4,
    "CD_BICLUSTER": 0.1,
    "CbG_TEXT": 0.193,
    "DaG_TEXT":0.1,
    "CtD_DB_TEXT": 0.3,
    "CtD_ALL": 0.814,
    "All_the_jawns": 0.146
}
final_dev_model = {}
final_test_model = {}
for model in tqdm_notebook(best_params):
    label_model.train(L[:, model_dict[model]], n_epochs=1000, verbose=False, lr=0.01, l2=best_params[model])
    final_dev_model[model] = label_model.predict_proba(L_dev[:, model_dict[model]])[:,0]
    final_test_model[model] = label_model.predict_proba(L_test[:, model_dict[model]])[:,0]


# In[ ]:


dev_marginals_df = pd.DataFrame.from_dict(final_dev_model)
dev_marginals_df.head(2)


# In[ ]:


test_marginals_df = pd.DataFrame.from_dict(final_test_model)
test_marginals_df.head(2)


# In[ ]:


model_aucs = plot_roc_curve(dev_marginals_df, candidate_dfs['dev'].curated_dsh, model_type='barplot',
                           xlim=[0.4,0.8])


# In[ ]:


get_auc_significant_stats(candidate_dfs['dev'], model_aucs)


# In[ ]:


model_aucs = plot_roc_curve(test_marginals_df, candidate_dfs['test'].curated_dsh, model_type='barplot',
                           xlim=[0.4,0.8])


# In[ ]:


get_auc_significant_stats(candidate_dfs['test'], model_aucs)

