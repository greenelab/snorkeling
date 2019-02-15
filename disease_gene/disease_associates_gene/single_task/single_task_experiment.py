
# coding: utf-8

# # Predicting Disease Associate Genes Relationship (Part 1)

# This notebook is designed to begin the process of predicting the disease associates gene (DaG) relationship. The first step this notebook takes is labeling our training sentences (split = 0) using our hand crafted label functions. Following the labeling process, the next step is to train a generative model to estimate the most likely training class ($\hat{Y}$) given the output of our label functions. From there the last step is to train word embedding vectors using facebook's [FastText](https://fasttext.cc/). These word embedding vectors will be used in part 2 where we train the discriminator models to make the final classification.

# ## Set up the environment

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import os
import pickle
import sys

sys.path.append(os.path.abspath('../../../modules'))

# Bayesian Optimization
from hyperopt import fmin, hp, tpe, Trials

from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, accuracy_score
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


from snorkel.learning.pytorch.rnn.rnn_base import mark_sentence
from snorkel.learning.pytorch.rnn.utils import candidate_to_tokens
from snorkel.models import Candidate, candidate_subclass

from metal.analysis import lf_summary
from metal.label_model import LabelModel
from metal.utils import plusminus_to_categorical

from gensim.models import FastText
from gensim.models import KeyedVectors

from utils.notebook_utils.label_matrix_helper import label_candidates, get_auc_significant_stats
from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.plot_helper import plot_curve, plot_label_matrix_heatmap

sys.path.append(os.path.abspath('data/label_functions'))
from disease_gene_lfs import DG_LFS


# In[4]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[5]:


quick_load = True


# ## Load the Data and Label the  Sentences

# This block of code is designed to label the sentences using our label functions. All the sentences are located in our postgres database that is store locally on the lab machine. The labeling process is defined as follows: Given a candidate id, we use the sqlalchemy library to extract a candidate object. Using this object and we pass it through a series of label functions that will output a 1 (positive), -1 (negative) or 0 (abstain) depending on the rule set. Lastly we aggregate the output of these functions into a sparse matrix that the generative model will use. Since these steps are pretty linear, we parallelized this process using python's multithreading library. Despite the optimization, this process can still take greater than 3 hours to label a set of ~600000 sentences.

# In[6]:


total_candidates_df = pd.read_table("../dataset_statistics/data/all_dg_candidates_map.tsv.xz")
total_candidates_df.head(2)


# In[7]:


spreadsheet_names = {
    'train': 'data/sentences/sentence_labels_train.xlsx',
    'dev': 'data/sentences/sentence_labels_dev.xlsx',
    'test': 'data/sentences/sentence_labels_test.xlsx'
}


# In[8]:


candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key])
    for key in spreadsheet_names
}

for key in candidate_dfs:
    print("Size of {} set: {}".format(key, candidate_dfs[key].shape[0]))


# In[9]:


#Save data for cluster
(
    candidate_dfs['dev'][["candidate_id", "curated_dsh"]]
    .to_csv("data/dev_labels.tsv", index=False, sep="\t")
)


# In[10]:


lfs = list(DG_LFS["DaG"].values())
lf_names = list(DG_LFS["DaG"].keys())


# In[11]:


if not quick_load:
    label_matricies = {
        'train':label_candidates(
            session, 
            (
                total_candidates_df
                .query("split==0&disease_mention_count==1&gene_mention_count==1")
                .candidate_id
                .values
                .tolist()
            ),
            lfs, 
            lf_names,
            num_threads=10,
            batch_size=50000,
            multitask=False
        )
    }


# In[12]:


if not quick_load:
    # Check to make sure the label functions match up with the candidate objects
    ids=label_matricies['train'].candidate_id.head(5).tolist()
    candidate_list = session.query(DiseaseGene).filter(DiseaseGene.id.in_(ids)).all()
    for candidate in tqdm_notebook(candidate_list):
        correct_output = list(map(lambda fn: fn(candidate), lfs))
        test_output = label_matricies['train'].query("candidate_id==@candidate.id").fillna(0).values.tolist()[0]
        for pair in zip(correct_output, test_output[:-1]):
            assert pair[0] == pair[1]


# In[13]:


if not quick_load:
    label_matricies.update({
        key:label_candidates(
            session, 
            candidate_dfs[key]['candidate_id'].values.tolist(),
            lfs, 
            lf_names,
            num_threads=10,
            batch_size=50000,
            multitask=False
        )
        for key in candidate_dfs
    })


# In[14]:


# Save the label matricies to a file for future loading/error analysis
if not quick_load:
    (
        label_matricies['train']
        .sort_values("candidate_id")
        .to_csv("data/train_sparse_matrix.tsv", sep="\t", index=False)
    )
    (
        label_matricies['dev']
        .sort_values("candidate_id")
        .to_csv("data/dev_sparse_matrix.tsv", sep="\t", index=False)
    )
    (
        label_matricies['test']
        .sort_values("candidate_id")
        .to_csv("data/test_sparse_matrix.tsv", sep="\t", index=False)
    )
# Quick load the label matricies
else:
    label_destinations = {
        'train':"data/train_sparse_matrix.tsv",
        'dev':"data/dev_sparse_matrix.tsv",
        'test':"data/test_sparse_matrix.tsv"
    }
    label_matricies = {
        key:pd.read_table(label_destinations[key]).to_sparse()
        for key in label_destinations
    }


# In[15]:


# Important Note Snorkel Metal uses a different coding scheme
# than the label functions output. (2 for negative instead of -1).
# This step corrects this problem by converting -1s to 2

train_ids = label_matricies['train'].candidate_id.isin(candidate_dfs['train'].candidate_id)

correct_L = plusminus_to_categorical(
    label_matricies['train']
    .sort_values("candidate_id")
    [train_ids == False]
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
)

correct_L_train = plusminus_to_categorical(
    label_matricies['train']
    .sort_values("candidate_id")
    [train_ids]
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
)

correct_L_dev = plusminus_to_categorical(
    label_matricies['dev']
    .sort_values("candidate_id")
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
)

correct_L_test = plusminus_to_categorical(
    label_matricies['test']
    .sort_values("candidate_id")
    .drop("candidate_id", axis=1)
    .to_coo()
    .toarray()
)


# In[16]:


lf_summary(
    sparse.coo_matrix(
        plusminus_to_categorical(
            label_matricies['train']
            .sort_values("candidate_id")
            .drop("candidate_id", axis=1)
            .to_coo()
            .toarray()
        )
    )
    .tocsr(), 
    lf_names=lf_names
)


# In[17]:


lf_summary(
    sparse.coo_matrix(
        plusminus_to_categorical(
            label_matricies['dev']
            .sort_values("candidate_id")
            .drop("candidate_id", axis=1)
            .to_coo()
            .toarray()
        )
    )
    .tocsr(), 
    lf_names=lf_names, 
    Y=candidate_dfs['dev'].curated_dsh.apply(lambda x: 1 if x> 0 else 2)
)


# The two above cells contain statistics for each label function used in this notebook. The generated statistics contains information as to the polarity of a label function, how often one label function overlaps with another, how ofter one label function conflicts with another, and many much does a given label function cover. If gold standard labels are provided, then extra information such as empirical accuracy can be incorporated. Upon insepction we have a wide range of performance in terms of individual label functions. Some are very accurate and some are almost completely wrong. Nonetheless both types of functions are important because the main goal here is to provide as much evidence as possible to estimate the tinal training class.

# ## Begin Training the Generative Model

# In this section, we are going to train a generative model. This model is designed to estimate the best class a training point belongs to given a set of label functions ($P(\hat{Y} \mid \lambda_{1\ldots n})$). This is accomplished by estimating a parameter called mu, which is the probability of a label function emitting a 1 or 2 given the true class. ($P(\lambda_{j} = y | Y=y)$). Once mu is estimated, the final step is to use mu and calculate the above probability ($P(\hat{Y} \mid \lambda_{1\ldots n})$). In order to find the best model, we will use hyperparameter optimization. This process involves trying multiple values keeping the value that provides the best result. The challenge with this process is that the search space is vast and can take non trivial time to navigate. As a consequence we resort to using advance methods called [bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) to navigate this space in an effective way. The hyperparameter we are tuning is the l2 norm. This penalty term penalizes a machine learning classifier for the magnitude of the weights. By using this parameter we optimize the l2 norm for each label function individually. Luckily, the generative model is pretty fast to train, so this process won't take a significant amount of time.

# # Distant Supervision

# Here in this section we are using the distant superivion paradigm to label our candidate sentences.

# ## Grid Search

# In[18]:


regularization_grid = pd.np.round(pd.np.linspace(0.01, 5, num=15), 2)


# In[19]:


grid_results = {}
label_model = LabelModel(k=2)
for param in tqdm_notebook(regularization_grid):
    label_model.train_model(correct_L[:,0:7], n_epochs=1000, print_every=200, seed=100, lr=0.01, l2=param)
    grid_results[str(param)] = label_model.predict_proba(correct_L_train[:,0:7])


# In[20]:


acc_results = defaultdict(list)

for key in grid_results:
    acc_results[key].append(
        accuracy_score(
            candidate_dfs['train']['curated_dsh'].fillna(0),
            list(map(lambda x: 1 if x > 0.5 else 0, grid_results[key][:,0]))
        )
    )
acc_df = pd.DataFrame(acc_results)
acc_df.head(2)


# In[21]:


acc_df.transpose().sort_values(0, ascending=False).head(5)


# In[22]:


plt.figure(figsize=(20,6))
plt.plot(list(acc_df.transpose().index), acc_df.transpose()[0],"bo-", label="DaG", )
plt.legend()


# In[23]:


label_model.train_model(correct_L[:, 0:7], n_epochs=1000, print_every=200, seed=100, lr=0.01, l2=1.08)
label_model.score(
    (correct_L_train[:, 0:7], candidate_dfs['train']['curated_dsh'].apply(lambda x: 1 if x > 0 else 2).values)
)


# In[24]:


lf_stats = zip(lf_names, range(0,label_model.mu.detach().clone().numpy().shape[0],2))
estimated_param = pd.np.clip(label_model.mu.detach().clone().numpy(), 0.01, 0.99)
value_type = ["P(L=1|Y=1)", "P(L=1|Y=2)", "P(L=2|Y=1)", "P(L=2|Y=2)"]
data = []

for lf_name, lf_index in lf_stats:
    data+=list(zip([lf_name] * len(value_type), estimated_param[lf_index:lf_index+2, :].flatten(), value_type))
    
label_function_weight_df = pd.DataFrame(data, columns=["label_function", "weight", "type"])
ax=sns.catplot(
    y="label_function", x="weight", col="type", 
    data=label_function_weight_df, orient="h", kind="bar",
    height=8, aspect=0.55
)


# In[25]:


dev_pred_ds_grid_df = pd.DataFrame(
    label_model.predict_proba(correct_L_dev[:, 0:7]),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['dev'].candidate_id.tolist())
dev_pred_ds_grid_df.head(2)


# In[26]:


test_pred_ds_grid_df = pd.DataFrame(
    label_model.predict_proba(correct_L_test[:,0:7]),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['test'].candidate_id.tolist())
test_pred_ds_grid_df.head(2)


# ## Bayesian Optimization

# In[27]:


def hyperparam_optimize(space, model, X, dev_X, dev_Y,lf_names, iterations=300):
    
    #Define the objective space to optimize
    def objective(params):
        #Randomly set up the parameters for the model to select
        l2=pd.np.fromiter(
            (params[key] for key in sorted(params.keys(), key=lambda x: int(x))),
            dtype=pd.np.float32
        )

        #Train the model
        model.train_model(
            X, n_epochs=1000, 
            print_every=200, seed=100, lr=0.01, 
            l2=l2, verbose=False
        )

        #Use accuracy score as the criteria
        score = model.score(
            (dev_X, dev_Y),
            verbose=False
        )
        # since search is design to minimize, change the sign
        return -score

    # Keep track of the trials
    trials = Trials()

    # Run the search!!
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, 
        max_evals=iterations, trials=trials, rstate=pd.np.random.RandomState(100)
    )
    return (
        pd.DataFrame(
            list(
                zip(
                    list(chain(*[(x+"_pos", x+"_neg") for x in lf_names])), 
                    [best[key] for key in sorted(best.keys(), key=lambda x: int(x))]
                )
            ), 
            columns=["label_function", "regularization"]
        ), 
        trials
    )


# In[28]:


search_space = {
        str(key):hp.uniform(str(key), 0, 10)
        for key in range(len(lf_names[0:7])*2)
    }
distant_supervision_reg_df, ds_trials = hyperparam_optimize(
    search_space, LabelModel(k=2),
    X=correct_L[:,0:7], dev_X=correct_L_train[:,0:7],
    dev_Y=candidate_dfs['train'].curated_dsh.apply(lambda x: 1 if x > 0 else 2).values,
    lf_names=lf_names[0:7], iterations=1000
)
distant_supervision_reg_df.head(2)


# In[29]:


ax = sns.barplot(y="label_function", x="regularization", data=distant_supervision_reg_df)
ax.set_title("L2 Param for Each Label Function")


# In[30]:


data = []
sampled_values = []
for trial in ds_trials.trials:
    data.append((trial['tid'], -trial['result']['loss']))
    sampled_values.append(list(map(lambda x: (trial['tid'], int(x[0]), x[1][0]), trial['misc']['vals'].items())))

sampled_values_df = pd.DataFrame(list(chain(*sampled_values)), columns=['iterations', 'param','value'])
trial_results_df = pd.DataFrame(data, columns=['iterations', 'acc'])
trial_results_df = (
    trial_results_df
    .assign(
        hue=trial_results_df.acc.apply(lambda x: True if x==trial_results_df.acc.max() else False).values
    )
)
trial_results_df.head(2)


# In[31]:


ax = sns.scatterplot(x='iterations', y='acc', hue='hue', data=trial_results_df, legend=None)
ax.set_title("Bayeisan Op Trial Results")


# In[32]:


g = sns.FacetGrid(sampled_values_df.sort_values("param"), col='param', height=4, aspect=0.9, col_wrap=5)
g.map(plt.scatter, "iterations", "value").add_legend()


# In[33]:


# Train best model
label_model = LabelModel(k=2)
label_model.train_model(
        correct_L[:, 0:7], n_epochs=1000, 
        print_every=200, seed=100, lr=0.01, 
        l2=distant_supervision_reg_df.regularization.values.astype(pd.np.float32)
    )
label_model.score(
    (correct_L_train[:, 0:7], candidate_dfs['train']['curated_dsh'].apply(lambda x: 1 if x > 0 else 2).values)
)


# In[34]:


lf_stats = zip(lf_names[0:7], range(0,label_model.mu.detach().clone().numpy().shape[0],2))
estimated_param = pd.np.clip(label_model.mu.detach().clone().numpy(), 0.01, 0.99)
value_type = ["P(L=1|Y=1)", "P(L=1|Y=2)", "P(L=2|Y=1)", "P(L=2|Y=2)"]
data = []

for lf_name, lf_index in lf_stats:
    data+=list(zip([lf_name] * len(value_type), estimated_param[lf_index:lf_index+2, :].flatten(), value_type))
    
label_function_weight_df = pd.DataFrame(data, columns=["label_function", "weight", "type"])
ax=sns.catplot(
    y="label_function", x="weight", col="type", 
    data=label_function_weight_df, orient="h", kind="bar",
    height=4, aspect=0.9
)


# The graph above shows the particular weights each label function has. Based on the polarity of each label function there are four categories: probability of a lf outputing a pos label given the true class is positive $P(L=1 \mid Y=1)$,  probability of a lf outputing a pos label given the true class is negative $P(L=1 \mid Y=2)$, probability of a lf outputing a neg label given the true class is positive $P(L=2 \mid Y=1)$ and probability of a lf outputting a neg label given the true class is negative $P(L=2 \mid Y=2)$. So far the weights look relatively logical; however, some of the positive label functions are providing evidence for the wrong class. This isn't a significant problem since the weights for wrong class are low, but this is a good sanity check in case undesired results arise. 

# In[35]:


dev_pred_ds_df = pd.DataFrame(
    label_model.predict_proba(correct_L_dev[:, 0:7]),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['dev'].candidate_id.tolist())
dev_pred_ds_df.head(2)


# In[36]:


test_pred_ds_df = pd.DataFrame(
    label_model.predict_proba(correct_L_test[:,0:7]),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['test'].candidate_id.tolist())
test_pred_ds_df.head(2)


# # Distant Supervision With Text Patterns

# Here in this section we incorporate more information such as text patterns, sentence context and other rules to sift through some of the noise caused by the general distant supervision approach.

# ## Grid Search

# In[37]:


regularization_grid = pd.np.round(pd.np.linspace(0.01, 1, num=15), 2)


# In[38]:


grid_results = {}
label_model = LabelModel(k=2)
for param in tqdm_notebook(regularization_grid):
    label_model.train_model(correct_L, n_epochs=1000, print_every=200, seed=100, lr=0.01, l2=param)
    grid_results[str(param)] = label_model.predict_proba(correct_L_train)


# In[39]:


acc_results = defaultdict(list)

for key in grid_results:
    acc_results[key].append(
        accuracy_score(
            candidate_dfs['train']['curated_dsh'].fillna(0),
            list(map(lambda x: 1 if x > 0.5 else 0, grid_results[key][:,0]))
        )
    )
acc_df = pd.DataFrame(acc_results)
acc_df.head(2)


# In[40]:


acc_df.transpose().sort_values(0, ascending=False).head(5)


# In[41]:


plt.figure(figsize=(20,6))
plt.plot(list(acc_df.transpose().index), acc_df.transpose()[0],"bo-", label="DaG", )
plt.legend()


# In[42]:


label_model.train_model(correct_L, n_epochs=1000, print_every=200, seed=100, lr=0.01, l2=0.01)
label_model.score(
    (correct_L_train, candidate_dfs['train']['curated_dsh'].apply(lambda x: 1 if x > 0 else 2).values)
)


# In[43]:


lf_stats = zip(lf_names, range(0,label_model.mu.detach().clone().numpy().shape[0],2))
estimated_param = pd.np.clip(label_model.mu.detach().clone().numpy(), 0.01, 0.99)
value_type = ["P(L=1|Y=1)", "P(L=1|Y=2)", "P(L=2|Y=1)", "P(L=2|Y=2)"]
data = []

for lf_name, lf_index in lf_stats:
    data+=list(zip([lf_name] * len(value_type), estimated_param[lf_index:lf_index+2, :].flatten(), value_type))
    
label_function_weight_df = pd.DataFrame(data, columns=["label_function", "weight", "type"])
ax=sns.catplot(
    y="label_function", x="weight", col="type", 
    data=label_function_weight_df, orient="h", kind="bar",
    height=8, aspect=0.55
)


# In[44]:


dev_pred_ds_txt_grid_df = pd.DataFrame(
    label_model.predict_proba(correct_L_dev),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['dev'].candidate_id.tolist())
dev_pred_ds_txt_grid_df.head(2)


# In[45]:


test_pred_ds_txt_grid_df = pd.DataFrame(
    label_model.predict_proba(correct_L_test),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['test'].candidate_id.tolist())
test_pred_ds_txt_grid_df.head(2)


# ## Bayesian Optimization

# In[46]:


search_space = {
        str(key):hp.uniform(str(key), 0, 10)
        for key in range(len(lf_names)*2)
    }
ds_text_reg_df, ds_text_trials = hyperparam_optimize(
    search_space, LabelModel(k=2),
    X=correct_L, dev_X=correct_L_train,
    dev_Y=candidate_dfs['train'].curated_dsh.apply(lambda x: 1 if x > 0 else 2).values,
    lf_names=lf_names, iterations=1000
)
ds_text_reg_df.head(2)


# In[47]:


fig, ax = plt.subplots()
fig.set_size_inches(19,20)
sns.barplot(y="label_function", x="regularization", data=ds_text_reg_df, ax=ax)
ax.set_title("L2 Param for Each Label Function")


# In[48]:


data = []
sampled_values = []
for trial in ds_text_trials.trials:
    data.append((trial['tid'], -trial['result']['loss']))
    sampled_values.append(list(map(lambda x: (trial['tid'], int(x[0]), x[1][0]), trial['misc']['vals'].items())))

sampled_values_df = pd.DataFrame(list(chain(*sampled_values)), columns=['iterations', 'param','value'])
trial_results_df = pd.DataFrame(data, columns=['iterations', 'acc'])
trial_results_df = (
    trial_results_df
    .assign(
        hue=trial_results_df.acc.apply(lambda x: True if x==trial_results_df.acc.max() else False).values
    )
)
trial_results_df.head(2)


# In[49]:


ax = sns.scatterplot(x='iterations', y='acc', hue='hue', data=trial_results_df, legend=None)
ax.set_title("Bayeisan Op Trial Results")


# In[50]:


g = sns.FacetGrid(sampled_values_df.sort_values("param"), col='param', height=4, aspect=0.9, col_wrap=5)
g.map(plt.scatter, "iterations", "value").add_legend()


# In[51]:


# Train best model
label_model = LabelModel(k=2)
label_model.train_model(
        correct_L, n_epochs=1000, 
        print_every=200, seed=100, lr=0.01, 
        l2=ds_text_reg_df.regularization.values.astype(pd.np.float32)
    )
label_model.score(
    (correct_L_train, candidate_dfs['train'].curated_dsh.apply(lambda x: 1 if x > 0 else 2).values)
)


# In[52]:


lf_stats = zip(lf_names, range(0,label_model.mu.detach().clone().numpy().shape[0],2))
estimated_param = pd.np.clip(label_model.mu.detach().clone().numpy(), 0.01, 0.99)
value_type = ["P(L=1|Y=1)", "P(L=1|Y=2)", "P(L=2|Y=1)", "P(L=2|Y=2)"]
data = []

for lf_name, lf_index in lf_stats:
    data+=list(zip([lf_name] * len(value_type), estimated_param[lf_index:lf_index+2, :].flatten(), value_type))
    
label_function_weight_df = pd.DataFrame(data, columns=["label_function", "weight", "type"])
ax=sns.catplot(
    y="label_function", x="weight", col="type", 
    data=label_function_weight_df, orient="h", kind="bar",
    height=8, aspect=0.55
)


# In[53]:


dev_pred_ds_txt_df = pd.DataFrame(
    label_model.predict_proba(correct_L_dev),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['dev'].candidate_id.tolist())
dev_pred_ds_txt_df.head(2)


# In[54]:


test_pred_ds_txt_df = pd.DataFrame(
    label_model.predict_proba(correct_L_test),
    columns=["pos_class_marginal", "neg_class_marginal"]
).assign(candidate_id=label_matricies['test'].candidate_id.tolist())
test_pred_ds_txt_df.head(2)


# # Model Performance

# In[55]:


aucs=plot_curve(
    (
        dev_pred_ds_df[["pos_class_marginal"]]
        .rename(index=str, columns={"pos_class_marginal":"Distant Supervision (DS)"})
        .assign(**{"DS_text_bayes":dev_pred_ds_txt_df.pos_class_marginal.values,
                  "DS_grid": dev_pred_ds_grid_df.pos_class_marginal.values,
                  "DS_text_grid": dev_pred_ds_txt_grid_df.pos_class_marginal.values})
    ),
    candidate_dfs['dev'].curated_dsh, 
    plot_title="Tune Set PRC", 
    metric="PR", model_type="curve"
)


# In[56]:


aucs=plot_curve(
    (
        dev_pred_ds_df[["pos_class_marginal"]]
        .rename(index=str, columns={"pos_class_marginal":"DS_bayes"})
        .assign(**{"DS_text_bayes":dev_pred_ds_txt_df.pos_class_marginal.values,
                  "DS_grid": dev_pred_ds_grid_df.pos_class_marginal.values,
                  "DS_text_grid": dev_pred_ds_txt_grid_df.pos_class_marginal.values})
    ),
    candidate_dfs['dev'].curated_dsh, 
    plot_title="Tune Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[57]:


get_auc_significant_stats(
    candidate_dfs['dev'],
    aucs
)


# In[58]:


aucs=plot_curve(
    (
        test_pred_ds_df[["pos_class_marginal"]]
        .rename(index=str, columns={"pos_class_marginal":"DS_bayes"})
        .assign(**{"DS_text_bayes":test_pred_ds_txt_df.pos_class_marginal.values,
                  "DS_grid": test_pred_ds_grid_df.pos_class_marginal.values,
                  "DS_text_grid": test_pred_ds_txt_grid_df.pos_class_marginal.values})
    ),
    candidate_dfs['test'].curated_dsh, 
    plot_title="Test Set PRC", 
    metric="PR", model_type="curve"
)


# In[59]:


aucs=plot_curve(
    (
        test_pred_ds_df[["pos_class_marginal"]]
        .rename(index=str, columns={"pos_class_marginal":"DS_bayes"})
        .assign(**{"DS_text_bayes":test_pred_ds_txt_df.pos_class_marginal.values,
                  "DS_grid": test_pred_ds_grid_df.pos_class_marginal.values,
                  "DS_text_grid": test_pred_ds_txt_grid_df.pos_class_marginal.values})
    ),
    candidate_dfs['test'].curated_dsh, 
    plot_title="Test Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[60]:


get_auc_significant_stats(
    candidate_dfs['test'],
    aucs
)


# In[61]:


grid_dist_df = (
    candidate_dfs['dev'][["candidate_id", "curated_dsh"]]
    .merge(dev_pred_ds_txt_grid_df[["candidate_id", "pos_class_marginal"]])
    .assign(outcome=lambda x: x.curated_dsh==x.pos_class_marginal.apply(lambda x: 1 if x > 0.5 else 0))
)
grid_dist_df.head(2)


# In[62]:


bayes_dist_df = (
    candidate_dfs['dev'][["candidate_id", "curated_dsh"]]
    .merge(dev_pred_ds_txt_df[["candidate_id", "pos_class_marginal"]])
    .assign(outcome=lambda x: x.curated_dsh==x.pos_class_marginal.apply(lambda x: 1 if x > 0.5 else 0))
)
bayes_dist_df.head(2)


# In[64]:


plt.hist(
    [
        bayes_dist_df.query("outcome==False").pos_class_marginal.values,
        grid_dist_df.query("outcome==False").pos_class_marginal.values
    ], 
    label=["bayes", "grid"]
)
plt.legend()
plt.xlabel("Likelihood of Positive Class")
plt.ylabel("Count")
plt.title("Distribution of False Predictions")


# ## Train Best Model and Output Marginals 

# In[65]:


label_model.train_model(
        correct_L, n_epochs=1000, 
        print_every=200, seed=100, lr=0.01, 
        l2=0.01
)

training_marginals_df = pd.DataFrame(
    label_model.predict_proba(
        plusminus_to_categorical(
            label_matricies['train']
            .sort_values("candidate_id")
            .drop("candidate_id", axis=1)
            .to_coo()
            .toarray()
        )
    ), 
    columns=["pos_class_marginal", "neg_class_marginal"]
)
training_marginals_df['candidate_id'] = (
    total_candidates_df
    .query("split==0&disease_mention_count==1&gene_mention_count==1")
    .candidate_id
    .sort_values()
    .values
)
training_marginals_df.to_csv("data/training_marginals.tsv", sep="\t", index=False)
training_marginals_df.head(2)


# In[66]:


ax = sns.distplot(training_marginals_df.pos_class_marginal, bins=10, kde=False, axlabel="Likelihood of Positive Class")
ax.set_title("Histogram of Predicted Likelihoods")


# In[67]:


spreadsheet_name = "data/sentence_gen_dev_error_analysis.xlsx"
writer = pd.ExcelWriter(spreadsheet_name)

(
    candidate_dfs['dev'][[
        "candidate_id", "disease", 
        "gene", "doid_id", "entrez_gene_id",
        "sentence_id", "sentence", "curated_dsh",
        "notes_dsh"
    ]]
    .assign(gen_model=dev_pred_ds_txt_df.pos_class_marginal.values)
    .to_excel(writer, sheet_name='sentences', index=False)
)

if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)

writer.close()


# # Word Vectors

# This notebook block contains word vectors to be used for the deep learning models. Using facebook's fasttext model, we trained a skipgram model using the sentences in the training set. Each word vector contains 300 dimensions. A validation check for these word embeddings is produced below with the top ten most similar words for diabetes.

# In[68]:


if os.path.isfile("data/training_word_vectors.bin"):
    model = KeyedVectors.load_word2vec_format("data/training_word_vectors.bin")
#Train a fasttext model
else:
    model = FastText(words_to_embed, window=2, negative=10, iter=50, sg=1, workers=4, alpha=0.005, size=300, seed=100)
    (
    model
    .wv
    .save_word2vec_format(
        "data/training_word_vectors.bin", 
        fvocab="data/training_word_vocab.txt", 
        binary=False
        )
    )


# In[69]:


model.most_similar("diabetes")


# In[70]:


from sklearn.manifold import TSNE
vocab = list(model.wv.vocab)
X = model[vocab]
tsne = TSNE(n_components=2)
tsne_df = pd.DataFrame(tsne.fit_transform(X), index=vocab, columns=["comp_1", "comp_2"])


# In[71]:


fig, ax = plt.subplots()
fig.set_size_inches(13,11)
sample_words = [
        "p53","women","men",
        "patient","genetic","positive",
        "chronic","cancer","carcinoma",
        "t2dm","diabetic","observed",
        "two","prognosis","function",
        "colon","vascular","low",
        "signaling","associated","association",
        "coronary","genotype","hla","ear",
        "methylation","inhibitors","chemotherapy",
        "allele","bladder","overexpression",
        "correlation","aim","study","conclusion",
        "schizophrenia","alzheimer","antibody",
        "myocardial","concentrations","hepatocellular",
        "objective","studied","significance","bone",
        "arthritis","depression","antigen","evaluated","anemia"
]
sns.scatterplot(x="comp_2", y="comp_1", data=tsne_df.loc[sample_words], ax=ax)
for word, pos in tsne_df.loc[sample_words].iterrows():
    ax.annotate(word, (pos['comp_2'], pos['comp_1']))
ax.set_title("t-SNE Plot of Generated Word Vectors")


# In[ ]:


word_dict_df = pd.DataFrame(
    [(word,idx) for idx, word in enumerate(model.vocab.keys(), 2)], 
    columns=["word", "index"]
)
word_dict_df = word_dict_df.append({"word":'*unknown*', "index":1}, ignore_index=True)
word_dict_df.to_csv("data/word_dictionary.tsv.xz", compression="xz", sep="\t", index=False)
word_dict = {word:index for word, index in word_dict_df[["word", "index"]].values}
word_dict_df.head(2)


# In[ ]:


def embed_word_to_index(cand):
    return [word_dict[word] if word in word_dict else 1 for word in cand]


# In[ ]:


def generate_embedded_df(candidates):
    words_to_embed = [
        (
        mark_sentence(
            candidate_to_tokens(cand), 
            [
                    (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                    (cand[1].get_word_start(), cand[1].get_word_end(), 2)
            ]
        ), cand.id)
        for cand in tqdm_notebook(candidates)
    ]
    embed_df = pd.DataFrame(list(map(lambda x: embed_word_to_index(x[0]), words_to_embed)))
    embed_df['candidate_id'] = list(map(lambda x: x[1], words_to_embed))
    embed_df['sen_length'] = list(map(lambda x: len(x[0]), words_to_embed))
    return embed_df


# In[ ]:


# Code to validate that the embedded rows align with the candidates
candidate_query = (
    total_candidates_df
    .query("split==0&disease_mention_count==1&gene_mention_count==1")
    .head(5)
    .candidate_id
    .astype(int)
    .tolist()
)
# Get the candidates
candidates = session.query(DiseaseGene).filter(DiseaseGene.id.in_(candidate_query)).all()

# Generate the correct output for each word
correct_words_to_embed=[
        (
        mark_sentence(
            candidate_to_tokens(cand), 
            [
                    (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                    (cand[1].get_word_start(), cand[1].get_word_end(), 2)
            ]
        ), cand.id)
        for cand in tqdm_notebook(candidates)
    ]

correct_embedded_words = list(map(lambda x: (embed_word_to_index(x[0]), x[1]), correct_words_to_embed))                            
embedded_train = generate_embedded_df(candidates)

# Confirm the dataframe contains the correct rows for each candidate
for words in correct_embedded_words:
    test_query = embedded_train.query("candidate_id==@words[1]").fillna(0).values.tolist()[0][:-2]
    for pair in zip(words[0], [col for col in test_query if col != 0]):
        assert pair[0] == pair[1]


# In[ ]:


training_data_df = generate_embedded_df(
    session
    .query(DiseaseGene)
    .filter(DiseaseGene.id.in_(
        total_candidates_df
        .query("split==0&disease_mention_count==1&gene_mention_count==1")
        .candidate_id
        .astype(int)
        .tolist()
    ))
    .all()
)
training_data_df.to_csv("data/training_dataframe.tsv.xz", sep="\t", compression="xz", index=False)
training_data_df.head(2)


# In[ ]:


dev_data_df = generate_embedded_df(
    session
    .query(DiseaseGene)
    .filter(DiseaseGene.id.in_(
        candidate_dfs['dev']['candidate_id']
        .values
        .tolist()
    ))
    .all()
)
dev_data_df.to_csv("data/dev_dataframe.tsv.xz", sep="\t", compression="xz", index=False)
dev_data_df.head(2)


# In[ ]:


test_data_df = generate_embedded_df(
    session
    .query(DiseaseGene)
    .filter(DiseaseGene.id.in_(
        candidate_dfs['test']['candidate_id']
        .values
        .tolist()
    ))
    .all()
)
test_data_df.to_csv("data/test_dataframe.tsv.xz", sep="\t", compression="xz", index=False)
test_data_df.head(2)

