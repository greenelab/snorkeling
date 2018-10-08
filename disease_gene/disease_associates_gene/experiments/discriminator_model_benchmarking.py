
# coding: utf-8

# # Discriminator Model Benchmarking

# The goal here is to find the best discriminator model for predicting disease associates gene (DaG) relationships. The few models tested in this are: bag of words, Doc2CecC 500k randomly sampled iterations, Doc2VecC all disease gene sentences, and a unidirectional long short term memory network (LSTM). The underlying hypothesis is that **The LSTM will be the best model in predicting DaG associations.**

# ## Set up The Environment

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import glob
from itertools import product
import pickle
import os
import sys

sys.path.append(os.path.abspath('../../../modules'))

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
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
from snorkel.learning.pytorch.rnn import LSTM
from snorkel.models import candidate_subclass, Candidate

from utils.label_functions import DG_LFS

from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.doc2vec_helper import get_candidate_objects, execute_doc2vec, write_sentences_to_file, run_grid_search
from utils.notebook_utils.label_matrix_helper import label_candidates, make_cids_query, get_auc_significant_stats
from utils.notebook_utils.train_model_helper import train_generative_model
from utils.notebook_utils.plot_helper import plot_curve


# In[4]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[5]:


quick_load = True


# # Get Estimated Training Labels

# From the work in the [previous notebook](gen_model_benchmarking.ipynb), we determined that the best parameters for the generative model are: 0.4 reg_param, 100 burnin interations and 100 epochs for training. Using this information, we trained the generative model to get the estimated training labels show in the historgram below.

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


gen_model = train_generative_model(
        label_matricies['train'],
        burn_in=100,
        epochs=100,
        reg_param=0.401,
        step_size=1/label_matricies['train'].shape[0],
        deps=DependencySelector().select(label_matricies['train']),
        lf_propensity=True
    )
training_prob_labels = gen_model.marginals(label_matricies['train'])
training_labels = list(map(lambda x: 1 if x > 0.5 else 0, training_prob_labels))


# In[10]:


import matplotlib.pyplot as plt
plt.hist(training_prob_labels)


# Based on this graph more than half of the data is receiving a positive label. Hard to tell if this is correct; however, based on some prior experience this seems to be incorrectly skewed towards the positive side. 

# ## Discriminator Models

# As mentioned above here we train various discriminator models to determine which model can best predict DaG sentences through noisy labels.

# ### Bag of Words Model

# In[11]:


vectorizer = CountVectorizer()
X = vectorizer.fit_transform(
    candidate_dfs['train'].sentence.values
)
dev_X = vectorizer.transform(candidate_dfs['dev'].sentence.values)
test_X = vectorizer.transform(candidate_dfs['test'].sentence.values)


# In[12]:


bow_model = run_grid_search(LogisticRegression(), X,  {'C':pd.np.linspace(1e-6,5, num=20)}, training_labels)


# In[13]:


plt.plot(pd.np.linspace(1e-6,5, num=20), bow_model.cv_results_['mean_train_score'])


# ### Doc2VecC

# This model comes from this [paper](https://arxiv.org/pdf/1707.02377.pdf), which is builds off of popular sentence/document embedding algorithms. Through their use of corruption, which involves removing words from a document to generate embeddings, the authors were able to achieve significant speed boosts and results. 
# Majority of the steps to embed these sentences are located in this script [here](../../generate_doc2vec_sentences.py). Shown below are results after feeding these embeddings into the logistic regression algorithm. 

# #### Doc2VecC 500k Subsample Experiment

# In[14]:


files = zip(
    glob.glob('../../doc2vec/doc_vectors/train_doc_vectors_500k_subset_*.txt'),
    glob.glob('../../doc2vec/doc_vectors/dev_doc_vectors_500k_subset_*.txt'),
    glob.glob('../../doc2vec/doc_vectors/test_doc_vectors_500k_subset_*.txt')
)

doc2vec_500k_dev_marginals_df = pd.DataFrame()
doc2vec_500k_test_marginals_df = pd.DataFrame()


# In[15]:


for index, data in tqdm_notebook(enumerate(files)):
    doc2vec_train = pd.read_table(data[0], header=None, sep=" ")
    doc2vec_train = doc2vec_train.values[:-1, :-1]
    
    doc2vec_dev = pd.read_table(data[1], header=None, sep=" ")
    doc2vec_dev = doc2vec_dev.values[:-1, :-1]
    
    doc2vec_test = pd.read_table(data[2], header=None, sep=" ")
    doc2vec_test = doc2vec_test.values[:-1, :-1]
    
    model = run_grid_search(LogisticRegression(), doc2vec_train,  
                            {'C':pd.np.linspace(1e-6, 5, num=4)}, training_labels)

    doc2vec_500k_dev_marginals_df['subset_{}'.format(index)] = model.predict_proba(doc2vec_dev)[:,1]
    doc2vec_500k_test_marginals_df['subset_{}'.format(index)] = model.predict_proba(doc2vec_test)[:,1]


# In[17]:


model_aucs=plot_curve(doc2vec_500k_dev_marginals_df, candidate_dfs['dev'].curated_dsh,
                      figsize=(20,6), model_type="scatterplot")


# In[18]:


doc2vec_subset_df = pd.DataFrame.from_dict(model_aucs, orient='index')
doc2vec_subset_df.describe()


# #### Doc2Vec All D-G Sentences

# In[19]:


doc2vec_X_all_DG = pd.read_table("../../doc2vec/doc_vectors/train_doc_vectors_all_dg.txt",
                             compression=None, header=None, sep=" ")
doc2vec_X_all_DG = doc2vec_X_all_DG.values[:-1,:-1]

doc2vec_dev_X_all_DG = pd.read_table("../../doc2vec/doc_vectors/dev_doc_vectors_all_dg.txt",
                             compression=None, header=None, sep=" ")
doc2vec_dev_X_all_DG = doc2vec_dev_X_all_DG.values[:-1,:-1]

doc2vec_test_X_all_DG = pd.read_table("../../doc2vec/doc_vectors/test_doc_vectors_all_dg.txt",
                             compression=None, header=None, sep=" ")
doc2vec_test_X_all_DG = doc2vec_test_X_all_DG.values[:-1,:-1]


# In[20]:


doc2vec_all_pubmed_model = run_grid_search(LogisticRegression(), doc2vec_X_all_DG,  
                                           {'C':pd.np.linspace(1e-6, 1, num=20)}, training_labels)


# In[21]:


plt.plot(pd.np.linspace(1e-6, 1, num=20), doc2vec_all_pubmed_model.cv_results_['mean_train_score'])


# ### LSTM 

# Here is the LSTM network uses the pytorch library. Because of the needed calculations, this whole sections gets ported onto penn's gpu cluster. Utilizing about 4 gpus this network takes less than a few hours to run depending on the embedding size.

# #### Train LSTM on GPU

# In[ ]:


lstm = LSTM()
cand_objs = get_candidate_objects(session, candidate_dfs)
X = lstm._preprocess_data(cand_objs['train'], extend=True)
dev_X = lstm.preprocess_data(cand_objs['dev'], extend=False)
test_X = lstm.preprocess_data(cand_objs['test'], extend=False)


# In[ ]:


pickle.dump(X, open('../../lstm_cluster/train_matrix.pkl', 'wb'))
pickle.dump(X, open('../../lstm_cluster/dev_matrix.pkl', 'wb'))
pickle.dump(X, open('../../lstm_cluster/test_matrix.pkl', 'wb'))
pickle.dump(lstm, open('../../lstm_cluster/model.pkl', 'wb'))
pickle.dump(training_labels, open('../../lstm_cluster/train_labels.pkl', 'wb'))
pickle.dump(candidate_dfs['dev'].curated_dsh.astype(int).tolist(), open('../../lstm_cluster/dev_labels.pkl', 'wb'))


# ### Look at LSTM Results

# In[22]:


dev_marginals = pickle.load(open('../../lstm_cluster/dev_lstm_marginals.pkl', 'rb'))
test_marginals = pickle.load(open('../../lstm_cluster/test_lstm_marginals.pkl', 'rb'))


# In[23]:


lstm_dev_marginals_df = pd.DataFrame.from_dict(dev_marginals)
lstm_test_marginals_df = pd.DataFrame.from_dict(test_marginals)


# In[24]:


model_aucs = plot_curve(lstm_dev_marginals_df, candidate_dfs['dev'].curated_dsh, model_type='heatmap',
                           y_label="Embedding Dim", x_label="Hidden Dim", metric="ROC")


# In[25]:


ci_auc_stats_df = get_auc_significant_stats(candidate_dfs['dev'], model_aucs).sort_values('auroc', ascending=False)
ci_auc_stats_df


# In[26]:


model_aucs = plot_curve(lstm_dev_marginals_df, candidate_dfs['dev'].curated_dsh, model_type='heatmap',
                           y_label="Embedding Dim", x_label="Hidden Dim", metric="PR")


# ### Let's see how the models compare with each other

# In[27]:


dev_marginals_df = pd.DataFrame(
    pd.np.array([
        gen_model.marginals(label_matricies['dev']),
        bow_model.predict_proba(dev_X)[:,1], 
        doc2vec_500k_dev_marginals_df['subset_2'],
        doc2vec_all_pubmed_model.predict_proba(doc2vec_dev_X_all_DG)[:,1],
        lstm_dev_marginals_df['1250,1000'].tolist()
    ]).T, 
    columns=['Gen_Model', 'Bag_of_Words', 'Doc2Vec 500k', 'Doc2Vec All DG', 'LSTM']
)
dev_marginals_df.head(2)


# In[28]:


model_aucs = plot_curve(
    dev_marginals_df, candidate_dfs['dev'].curated_dsh, 
    plot_title="Dev ROC", model_type='curve', 
    figsize=(10,6), metric="ROC"
)


# In[29]:


get_auc_significant_stats(candidate_dfs['dev'], model_aucs)


# In[30]:


test_marginals_df = pd.DataFrame(
    pd.np.array([
        gen_model.marginals(label_matricies['test']),
        bow_model.best_estimator_.predict_proba(test_X)[:,1],
        doc2vec_500k_test_marginals_df['subset_2'],
        doc2vec_all_pubmed_model.best_estimator_.predict_proba(doc2vec_test_X_all_DG)[:,1],
        lstm_test_marginals_df['1250,1000'].tolist()
    ]).T, 
    columns=['Gen_Model', 'Bag_of_Words', 'Doc2Vec 500k', 'Doc2Vec All DG', 'LSTM']
)
test_marginals_df.head(2)


# In[31]:


model_aucs = plot_curve(
    test_marginals_df, candidate_dfs['test'].curated_dsh, 
    plot_title="Test ROC", model_type='curve', 
    figsize=(10,6), metric="ROC"
)


# In[32]:


get_auc_significant_stats(candidate_dfs['test'], model_aucs)


# In[33]:


model_aucs = plot_curve(
    test_marginals_df, candidate_dfs['test'].curated_dsh, 
    plot_title="Test PRC", model_type='curve', 
    figsize=(10,6), metric="PR"
)

