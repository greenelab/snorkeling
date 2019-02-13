
# coding: utf-8

# # Predicting Disease Associate Genes Relationship (Part 2)

# This notebook is designed to analyze the discriminator models used for this project. Based on the power of weak supervision, we now have access to deep learning models such as long short term memory networks (LSTM) or convolutional neural networks (CNN). This is exciting because, previous research studies have shown great success using these models; however, a significant caveat is that these classifiers are difficult to train. Usually they will overfit to the training dataset, which leads towards poor performance on the classification task.

# ## Set Up the Environment

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


from collections import OrderedDict
import glob
import os
import pickle
import re
import sys

sys.path.append(os.path.abspath('../../../modules'))
sys.path.append(os.path.abspath('../'))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook

import torch
import torch.nn as nn


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


from snorkel.models import Candidate, candidate_subclass
from cnn import PaddedEmbeddings, CNN
from metal.modules import LSTMModule
from metal.modules.lstm_module import EmbeddingsEncoder
from metal.end_model import EndModel

from utils.notebook_utils.label_matrix_helper import label_candidates, get_auc_significant_stats
from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.plot_helper import plot_curve, plot_label_matrix_heatmap
from utils.notebook_utils.train_model_helper import get_attn_scores, get_network_results


# In[4]:


DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[5]:


spreadsheet_names = {
    #'train': 'multitask_experiment/data/sentences/sentence_labels_train.xlsx',
    'dev': 'data/sentences/sentence_labels_dev.xlsx',
    'test': 'data/sentences/sentence_labels_test.xlsx',
}


# In[6]:


candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key])
    for key in spreadsheet_names
}

for key in candidate_dfs:
    print("Size of {} set: {}".format(key, candidate_dfs[key].shape[0]))


# In[7]:


dg_map_df = pd.read_table("../dataset_statistics/all_dg_candidates_map.tsv.xz")
dg_map_df.head(2)


# In[8]:


dev_data = pd.read_table("data/dev_dataframe.tsv.xz").sort_values("candidate_id")
dev_data.head(2)


# In[9]:


test_data = pd.read_table("data/test_dataframe.tsv.xz").sort_values("candidate_id")
test_data.head(2)


# In[10]:


word_vectors = pd.read_table(
    "data/training_word_vectors.bin", 
    sep=" ", skiprows=1,
    header=None,index_col=0, 
    keep_default_na=False
)
word_vectors.head(2)


# In[11]:


cutoff = 60


# In[12]:


dev_X = torch.LongTensor(
    dev_data
    .query("sen_length < @cutoff")
    [[col for col in dev_data.columns if col in list(map(lambda x: str(x),range(cutoff-1)))]]
    .fillna(0)
    .values
)
dev_Y = torch.LongTensor(
    dev_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['dev'])
    .curated_dsh
    .apply(lambda x: 1 if x > 0 else 2)
    .values
)

test_X = torch.LongTensor(
    test_data
    .query("sen_length < @cutoff")
    [[col for col in test_data.columns if col in list(map(lambda x: str(x),range(cutoff-1)))]]
    .fillna(0)
    .values
)
test_Y = torch.LongTensor(
    test_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['test'])
    .curated_dsh
    .apply(lambda x: 1 if x > 0 else 2)
    .values
)


# In[13]:


gen_model_dev_df = (
    pd.read_table("data/gen_model_dev_set_pred.tsv")
    .merge(dev_data.query("sen_length < @cutoff"), on="candidate_id")
    .sort_values("candidate_id")
)
gen_model_test_df = (
    pd.read_table("data/gen_model_test_set_pred.tsv")
    .merge(test_data.query("sen_length < @cutoff"), on="candidate_id")
    .sort_values("candidate_id")
)


# # LSTM Network Evaluation

# Used a LSTM network with an attention layer at the end. The following parameters for the network are produced below in the table:
# 
# | Parameter | Network 1 | Network 2 | Network 3 | Network 4 | Network 5 | Network 6 |
# |-------|-------|-------|-------|-------|-------|-------|
# | Word Embeddings | 300 dim (fixed) | 300 dim (fixed) | 300 dim (fixed) | 300 dim (fixed) | 300 dim (free) | 300 dim (free) | 
# | Hidden State | 50 Dim |  100 Dim | 250 Dim | 300 Dim | 250 Dim | 250 Dim |
# | Dropout | 0.25 (outside) and 0.25 (inside) | 0.25 (outside) and 0.25 (inside) | 0.25 (outside) and 0.25 (inside) | 0.25 (outside) and 0.25 (inside) | 0.25 (outside) and 0.25 (inside) | 0.25 (outside) and 0.25 (inside) |
# | Layers | 2 | 2 | 2 | 2 | 2 | 2 |
# | learning rate | 0.01 |  0.01 |  0.01 |  0.01 | 0.01 | 0.01 | 
# | optimizer | adam with betas (0.9, 0.99) | adam with betas (0.9, 0.99) | adam with betas (0.9, 0.99) | adam with betas (0.9, 0.99) | adam with betas (0.9, 0.99) | adam with betas (0.9, 0.99) |
# | Batch Size | 256 | 256 | 256 | 256 | 256 | 64 | 
#     

# In[40]:


lstm_params = {
    "LSTM Network 1":
    {
        "hidden_size":50,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
    },
    "LSTM Network 2":
    {
        "hidden_size":100,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
    },
    "LSTM Network 3":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
    },
    "LSTM Network 4":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
    },
    "LSTM Network 5":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.01
    },
    "LSTM Network 6":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.1
    },
    "LSTM Network 7":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.25
    },
    "LSTM Network 8":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.35,
        "batch_norm":False,
    },
    "LSTM Network 9":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.45,
        "batch_norm":False,
    },
    "LSTM Network 10":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.5
    },
    "LSTM Network 11":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.05,
        "batch_norm":True,
    },
    "LSTM Network 12":
    {
        "hidden_size":250,
        "output_size":2,
        "seed":100,
        "word_embeddings":300,
        "vocab_size":word_vectors.shape[0]+2,
        "max_dim": 59,
        "output_size": 2,
        "num_layers": 2,
        "freeze_embeddings":True,
        "outside_dropout":0.25,
        "inside_dropout": 0.25,
        "l2":0.25,
        "batch_norm":True,
    },

}


# In[41]:


lstm_model_paths = {
    "LSTM Network 1":sorted(
            glob.glob("data/final_models/fixed_index/lstm/300_50_frozen_both_dropout_0.25/*checkpoint*"), 
            key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 2":sorted(
            glob.glob("data/final_models/fixed_index/lstm/300_100_frozen_both_dropout_0.25/*checkpoint*"),
            key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 3":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 4":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_300_frozen_both_dropout_0.25/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 5":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.01/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 6":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.1/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 7":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.25/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 8":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.35/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 9":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.45/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 10":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.5/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 11":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.05_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "LSTM Network 12":sorted(
        glob.glob("data/final_models/fixed_index/lstm/300_250_frozen_both_dropout_0.25_l2_0.25_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    )
}


# In[42]:


lstm_end_models = {}


# In[43]:


for key in lstm_params:
    lstm_end_models[key] = EndModel(
        [lstm_params[key]["max_dim"], lstm_params[key]["hidden_size"]*2, lstm_params[key]["output_size"]], 
        middle_modules=[LSTMModule(
                encoded_size=lstm_params[key]["word_embeddings"],
                hidden_size=lstm_params[key]["hidden_size"],
                lstm_reduction='attention',
                lstm_num_layers=lstm_params[key]['num_layers'],
                encoder_class=EmbeddingsEncoder,
                encoder_kwargs={
                "vocab_size":lstm_params[key]["vocab_size"],
                "freeze":lstm_params[key]["freeze_embeddings"],
                "seed":lstm_params[key]["seed"],
                }
                )],
        seed=lstm_params[key]["seed"], 
        use_cuda=False,
        middle_layer_config = {
        'middle_relu':False,
        'middle_dropout': lstm_params[key]["outside_dropout"],
        'middle_batchnorm': lstm_params[key]["batch_norm"] if "batch_norm" in lstm_params[key] else False
        },
    )    


# In[44]:


lstm_results = {}
for network in lstm_model_paths:
    lstm_results[network] = get_network_results(
        lstm_model_paths[network], lstm_end_models[network], 
        dev_X, test_X
    )


# In[45]:


lstm_results[network][0].head(2)


# In[46]:


lstm_results[network][1].head(2)


# In[47]:


lstm_results[network][2].head(2)


# In[48]:


lstm_results[network][3]


# # LSTM Dimension Size

# In[49]:


model_results = [
    lstm_results["LSTM Network 1"][0], lstm_results["LSTM Network 2"][0], 
    lstm_results["LSTM Network 3"][0], lstm_results["LSTM Network 4"][0],
    lstm_results["LSTM Network 5"][0], lstm_results["LSTM Network 6"][0],
    lstm_results["LSTM Network 7"][0], lstm_results["LSTM Network 8"][0], 
    lstm_results["LSTM Network 9"][0], lstm_results["LSTM Network 10"][0], 
    lstm_results["LSTM Network 11"][0], lstm_results["LSTM Network 12"][0],
]
labels = [
    "Hidden Dim (50)", "Hidden Dim (100)", 
    "Hidden Dim (250)", "Hidden Dim (300)",
    "Hidden Dim (250) L2: (0.01)", "Hidden Dim (250) L2: (0.1)",
    "Hidden Dim (250) L2: (0.25)", "Hidden Dim (250) L2: (0.35)", 
    "Hidden Dim (250) L2: (0.45)", "Hidden Dim (250) L2: (0.5)", 
    "Hidden Dim (250) L2: (0.05) Batch", "Hidden Dim (250) L2: (0.25) Batch"
]

plt.rcParams.update({'font.size':15})

fig, axn = plt.subplots(3,4, sharex=True, sharey=True)
fig.set_size_inches((20,9))
fig.suptitle("Learning Curve")
for i, (ax, data, plot_title) in enumerate(zip(axn.flat, model_results, labels)):
    l1, l2 = ax.plot(data['epoch'], data["train_loss"], data['epoch'], data["val_loss"])
    ax.set_title(plot_title)
    if i == 0:
        fig.legend((l1, l2), ("train", "val"), 'center right')
fig.text(0.5, 0.04, 'Epochs', ha='center')
fig.text(0.04, 0.5, 'Cross Entropy Loss', va='center', rotation='vertical')


# In[51]:


lstm_dev_hidden_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            lstm_results["LSTM Network 1"][1][ lstm_results["LSTM Network 1"][3]].values,
            lstm_results["LSTM Network 2"][1][ lstm_results["LSTM Network 2"][3]].values,
            lstm_results["LSTM Network 3"][1][ lstm_results["LSTM Network 3"][3]].values,
            lstm_results["LSTM Network 4"][1][ lstm_results["LSTM Network 4"][3]].values,
        ],
            axis=1), 
        columns=[
            "Gen_Model", "Bi-LSTM (50)", 
            "Bi-LSTM (100)","Bi-LSTM (250)", 
            "Bi-LSTM(300)", 
                ]
    )
aucs=plot_curve(
    lstm_dev_hidden_df,
    dev_Y.numpy(), 
    plot_title="Tune Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[67]:


lstm_dev_l2_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            lstm_results["LSTM Network 5"][1][ lstm_results["LSTM Network 5"][3]].values,
            lstm_results["LSTM Network 6"][1][ lstm_results["LSTM Network 6"][3]].values,
            lstm_results["LSTM Network 7"][1][ lstm_results["LSTM Network 7"][3]].values,
            lstm_results["LSTM Network 8"][1][ lstm_results["LSTM Network 8"][3]].values,
            lstm_results["LSTM Network 9"][1][ lstm_results["LSTM Network 9"][3]].values,
            lstm_results["LSTM Network 10"][1][ lstm_results["LSTM Network 10"][3]].values
        ],
            axis=1), 
        columns=["Gen_Model", "Bi-LSTM (250) L2: (0.01)", 
                 "Bi-LSTM (250) L2: (0.1)","Bi-LSTM (250) L2: (0.25)",
                 "Bi-LSTM (250) L2: (0.35)","Bi-LSTM (250) L2: (0.45)", 
                 "Bi-LSTM (250) L2: (0.5)"
                ]
    )
aucs=plot_curve(
    lstm_dev_l2_df,
    dev_Y.numpy(), 
    plot_title="Tune Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[62]:


lstm_dev_batch_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            lstm_results["LSTM Network 11"][1][ lstm_results["LSTM Network 11"][3]].values,
            lstm_results["LSTM Network 12"][1][ lstm_results["LSTM Network 12"][3]].values
        ],
            axis=1), 
        columns=["Gen_Model",  
                 "Bi-LSTM (250) L2: (0.05) Batch", 
                 "Bi-LSTM (250) L2: (0.25) Batch"
                ]
    )
aucs=plot_curve(
    lstm_dev_batch_df,
    dev_Y.numpy(), 
    plot_title="Tune Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[ ]:


get_auc_significant_stats(
     dev_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['dev']),
    aucs
)


# In[56]:


lstm_test_df = pd.DataFrame(
        pd.np.stack([
            gen_model_test_df["gen_model"].values,
            lstm_results["LSTM Network 1"][2][lstm_results["LSTM Network 1"][3]].values,
            lstm_results["LSTM Network 2"][2][lstm_results["LSTM Network 2"][3]].values,
            lstm_results["LSTM Network 3"][2][lstm_results["LSTM Network 3"][3]].values,
            lstm_results["LSTM Network 4"][2][lstm_results["LSTM Network 4"][3]].values,
        ],
            axis=1), 
        columns=["Gen_Model", "Bi-LSTM (50)", "Bi-LSTM (100)","Bi-LSTM (250)", "Bi-LSTM (300)", 
                ]
    )
aucs=plot_curve(
    lstm_test_df,
    test_Y.numpy(), 
    plot_title="Test Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[59]:


lstm_test_df = pd.DataFrame(
        pd.np.stack([
            gen_model_test_df["gen_model"].values,
            lstm_results["LSTM Network 5"][2][lstm_results["LSTM Network 5"][3]].values,
            lstm_results["LSTM Network 6"][2][lstm_results["LSTM Network 6"][3]].values,
            lstm_results["LSTM Network 7"][2][ lstm_results["LSTM Network 7"][3]].values,
            lstm_results["LSTM Network 8"][2][ lstm_results["LSTM Network 8"][3]].values,
            lstm_results["LSTM Network 9"][2][ lstm_results["LSTM Network 9"][3]].values,
            lstm_results["LSTM Network 10"][2][ lstm_results["LSTM Network 10"][3]].values,
        ],
            axis=1), 
        columns=["Gen_Model",
                 "Bi-LSTM (250) L2: (0.01)", "Bi-LSTM(250) L2: (0.1)",
                 "Bi-LSTM (250) L2: (0.25)", "Bi-LSTM (250) L2: (0.35)", 
                 "Bi-LSTM (250) L2: (0.45)", "Bi-LSTM (250) L2: (0.5)", 
                ]
    )
aucs=plot_curve(
    lstm_test_df,
    test_Y.numpy(), 
    plot_title="Test Set ROC", 
    metric="PR", 
    model_type="curve"
)


# In[68]:


lstm_test_batch_df = pd.DataFrame(
        pd.np.stack([
            gen_model_test_df["gen_model"].values,
            lstm_results["LSTM Network 11"][2][ lstm_results["LSTM Network 11"][3]].values,
            lstm_results["LSTM Network 12"][2][ lstm_results["LSTM Network 12"][3]].values,
        ],
            axis=1), 
        columns=["Gen_Model",
                 "Bi-LSTM (250) L2: (0.05) Batch", 
                 "Bi-LSTM (250) L2: (0.25) Batch", 
                ]
    )
aucs=plot_curve(
    lstm_test_batch_df,
    test_Y.numpy(), 
    plot_title="Test Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[80]:


lstm_test_batch_df = pd.DataFrame(
        pd.np.stack([
            gen_model_test_df["gen_model"].values,
            lstm_results["LSTM Network 11"][2][ lstm_results["LSTM Network 11"][3]].values,
            lstm_results["LSTM Network 12"][2][ lstm_results["LSTM Network 12"][3]].values,
        ],
            axis=1), 
        columns=["Gen_Model",
                 "Bi-LSTM (250) L2: (0.05) Batch", 
                 "Bi-LSTM (250) L2: (0.25) Batch", 
                ]
    )
aucs=plot_curve(
    lstm_test_batch_df,
    test_Y.numpy(), 
    plot_title="Test Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[81]:


get_auc_significant_stats(
     test_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['test']),
    aucs
)


# # Visualize LSTM Attention Layer

# The attention layer is a useful tool, because it is a linear combination of all the hidden states that are outputted from an LSTM network or any given recurrent neural network. Based on this fact, the hidden state that receives a higher weight means that word contributes more towards the final output than the other hidden states. Below in this network are heatmaps that show what words our LSTM network is paying close attention to and which words it is ignoring.

# In[69]:


word_dict_df = pd.read_table("data/word_dictionary.tsv.xz")
reverse_dict = {index:word for word, index in word_dict_df[["word", "index"]].values}
word_dict_df.head(2)


# ### Positive Sentence Example

# In[70]:


index = 422
words = [reverse_dict[col.item()] for col in dev_X[index] if col > 0]
attn_df_dict = get_attn_scores(lstm_model_paths, lstm_end_models, dev_X[index:index+1], words)


# In[76]:


print(" ".join(words))
print()
print("P(Y|X) = {:.2f}".format(lstm_results["LSTM Network 10"][1][lstm_results["LSTM Network 10"][3]].iloc[index]))
print("True Y = {}".format(dev_Y[index]))


# In[73]:


plt.rcParams.update({'font.size':15})
fig, axn = plt.subplots(1, 12, sharex=True, sharey=True)
fig.set_size_inches((25,18))
fig.suptitle("Visualization of Attention Layer")
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, (ax, key) in enumerate(zip(axn.flat, attn_df_dict.keys())):
    sns.heatmap(
        attn_df_dict[key].set_index("words"),
        annot=False,
        cmap='viridis',
        xticklabels=False,
        ax=ax,
        cbar_ax = None if i else cbar_ax,
        cbar = (i==0)
    )
    ax.set_title(key)


# ### Negative Sentence Example

# In[143]:


index = 15
words = [reverse_dict[col.item()] for col in dev_X[index] if col > 0]
attn_df_dict = get_attn_scores(lstm_model_paths, lstm_end_models, dev_X[index:index+1], words)


# In[144]:


print(" ".join(words))
print()
print("P(Y|X) = {:.2f}".format(lstm_results["LSTM Network 12"][1][lstm_results["LSTM Network 12"][3]].iloc[index]))
print("True Y = {}".format(dev_Y[index]))


# In[145]:


plt.rcParams.update({'font.size':15})
fig, axn = plt.subplots(1, 12, sharex=True, sharey=True)
fig.set_size_inches((25,18))
fig.suptitle("Visualization of Attention Layer")
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, (ax, key) in enumerate(zip(axn.flat, attn_df_dict.keys())):
    sns.heatmap(
        attn_df_dict[key].set_index("words"),
        annot=False,
        cmap='viridis',
        xticklabels=False,
        yticklabels = True,
        ax=ax,
        cbar_ax = None if i else cbar_ax,
        cbar = (i==0)
    )
    ax.set_title(key)


# In[85]:


test_sentence = ["the", "gene",  "~~[[2", "brca2", "2]]~~", "is", "not", "expressed", "in", "~~[[1", "cancer","1]]~~","."]
test_data_point = torch.LongTensor([list(map(lambda x: word_dict_df.query("word==@x")['index'].values[0], test_sentence))])
attn_df_dict = get_attn_scores(lstm_model_paths, lstm_end_models, test_data_point, test_sentence)


# In[86]:


print(" ".join(test_sentence))
print()
print("P(Y|X) = {:.2f}".format(lstm_end_models["LSTM Network 3"].predict_proba(test_data_point)[:,0][0]))
print("True Y = {}".format(1))


# In[87]:


plt.rcParams.update({'font.size':15})
fig, axn = plt.subplots(1, 12, sharex=True, sharey=True)
fig.set_size_inches((25,18))
fig.suptitle("Visualization of Attention Layer")
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, (ax, key) in enumerate(zip(axn.flat, attn_df_dict.keys())):
    sns.heatmap(
        attn_df_dict[key].set_index("words"),
        annot=False,
        cmap='viridis',
        xticklabels=False,
        yticklabels = True,
        ax=ax,
        cbar_ax = None if i else cbar_ax,
        cbar = (i==0)
    )
    ax.set_title(key)
    torch.LongTensor([list(map(lambda x: word_dict_df.query("word==@x")['index'].values[0], ["the", "gene", "brca2", "is", "not", "associated", "with", "cancer"]))])


# In[88]:


test_sentence = ["the", "gene", "~~[[2", "brca2","2]]~~", "is", "highly", "expressed", "in", "~~[[1", "cancer","1]]~~", "."]
test_data_point = torch.LongTensor([list(map(lambda x: word_dict_df.query("word==@x")['index'].values[0], test_sentence))])
attn_df_dict = get_attn_scores(lstm_model_paths, lstm_end_models, test_data_point, test_sentence)


# In[92]:


print(" ".join(test_sentence))
print()
print("P(Y|X) = {:.2f}".format(lstm_end_models["LSTM Network 11"].predict_proba(test_data_point)[:,0][0]))
print("True Y = {}".format(1))


# In[90]:


plt.rcParams.update({'font.size':15})
fig, axn = plt.subplots(1, 9, sharex=True, sharey=True)
fig.set_size_inches((25,18))
fig.suptitle("Visualization of Attention Layer")
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, (ax, key) in enumerate(zip(axn.flat, attn_df_dict.keys())):
    sns.heatmap(
        attn_df_dict[key].set_index("words"),
        annot=False,
        cmap='viridis',
        xticklabels=False,
        yticklabels = True,
        ax=ax,
        cbar_ax = None if i else cbar_ax,
        cbar = (i==0)
    )
    ax.set_title(key)
    torch.LongTensor([list(map(lambda x: word_dict_df.query("word==@x")['index'].values[0], ["the", "gene", "brca2", "is", "not", "associated", "with", "cancer"]))])


# # CNN Network Evaluation

# Used a Convolutional neural network with two fully connected layers at the end. The following parameters for this network are produced below in the table:
# 
# | Parameter | Network 1 | Network 2 | Network 3 | Network 4 | Network 5 |
# |-------|-------|-------|-------|-------|-------|
# | Word Embeddings | 300 dim (fixed) | 300 dim (fixed) | 300 dim (fixed) | 300 dim (fixed) | 300 dim (free) |
# | kernel Sizes | 7,7,7,7 | 7,7,7,7 | 3,4,5,6 | 10,10,10,10 | 7,7,7,7 |
# | Batch Norm | Yes | No | Yes | Yes | Yes |
# | Dropout | 0.5 (outside) | 0.5 (outside) | 0.5 (outside) | 0.5 (outside) | 0.5 (outside) | 
# | Layers | 2 | 2 | 2 | 2 | 2 | 
# | learning rate | 0.0001 |  0.0001 |  0.0001 |  0.0001 | 0.0001 | 
# | optimizer | adam with betas(0.9, 0.99)| adam with betas(0.9, 0.99)| adam with betas(0.9, 0.99)| adam with betas(0.9, 0.99)| adam with betas(0.9, 0.99)|
# | Batch Size | 256 | 256 | 256 | 256 | 256 |
#     

# In[93]:


cnn_params = {
    "CNN Network 1":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True
    },
    "CNN Network 2":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":False
    },
    "CNN Network 3":{
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[3,4,5,6],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True
    },
    "CNN Network 4":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[10,10,10,10],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True
    },
    "CNN Network 5":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True
    },
    "CNN Network 6":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True,
        "l2":0.5
    },
    "CNN Network 7":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True,
        "l2":2
    },
    "CNN Network 8":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True,
        "l2":3.5
    },
    "CNN Network 9":
    {
        "hidden_size":300,
        "output_size":2,
        "seed":100,
        "kernel_sizes":[7,7,7,7],
        "vocab_size":word_vectors.shape[0]+2,
        "batchnorm":True,
        "l2":5
    }
}


# In[94]:


cnn_model_paths = {
    "CNN Network 1":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_frozen_0.0001_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 2":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_frozen_0.0001_no_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 3":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_3456_frozen_0.0001_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 4":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_10101010_frozen_0.0001_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 5":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_free_0.0001_batch_norm/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 6":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_frozen_0.0001_batch_norm_0.5/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 7":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_frozen_0.0001_batch_norm_2/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 8":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_frozen_0.0001_batch_norm_3.5/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
    "CNN Network 9":sorted(
        glob.glob("data/final_models/fixed_index/cnn/100_4_7777_frozen_0.0001_batch_norm_5/*checkpoint*"),
        key=lambda x: int(re.search(r'([\d]+)', os.path.basename(x)).group())
    ),
}


# In[95]:


cnn_end_models= {}


# In[96]:


for network in cnn_params:
    cnn_end_models[network] = EndModel(
    [cutoff-1, 50, 2], 
    input_module=PaddedEmbeddings(
        cnn_params[network]["vocab_size"], cnn_params[network]["hidden_size"], freeze=True
        ),
    middle_modules=[CNN(100, cnn_params[network]["kernel_sizes"], 59, 300)],
    seed=cnn_params[network]["seed"], 
    use_cuda=False,
    middle_layer_config = {
    'middle_relu':False,
    'middle_dropout': 0.25,
    'middle_batchnorm':cnn_params[network]["batchnorm"],
    },
    input_layer_config = {
        'input_relu':False,
        'input_batchnorm':False,
        'input_dropout': 0,
    }
)


# In[ ]:


cnn_results = {}
for network in cnn_model_paths:
    cnn_results[network] = get_network_results(
        cnn_model_paths[network], cnn_end_models[network], 
        dev_X, test_X
    )


# In[102]:


cnn_results[network][0].head(2)


# In[103]:


cnn_results[network][1].head(2)


# In[104]:


cnn_results[network][2].head(2)


# In[105]:


cnn_results[network][3]


# In[106]:


cnn_model_results = [
    cnn_results["CNN Network 1"][0],cnn_results["CNN Network 2"][0],
    cnn_results["CNN Network 3"][0], cnn_results["CNN Network 4"][0],
    cnn_results["CNN Network 5"][0], cnn_results["CNN Network 6"][0],
    cnn_results["CNN Network 7"][0], cnn_results["CNN Network 8"][0],
    cnn_results["CNN Network 9"][0]
]
cnn_labels = [
    "CNN (7,7,7,7 batch)", "CNN (7,7,7,7)", 
    "CNN (3,4,5,6)", "CNN (10,10,10,10)",
    "CNN (7,7,7,7 free)", "CNN (7,7,7,7) L2:0.5", 
    "CNN (7,7,7,7) L2:2", "CNN (7,7,7,7) L2:3.5", 
    "CNN (7,7,7,7) L2:5"
]

plt.rcParams.update({'font.size':18})

fig, axn = plt.subplots(3, 3, sharex=True, sharey=True)
fig.set_size_inches((18,7))
fig.suptitle("Learning Curve")
for i, (ax, data, plot_title) in enumerate(zip(axn.flat, cnn_model_results, cnn_labels)):
    l1, l2 = ax.plot(data['epoch'], data["train_loss"], data['epoch'], data["val_loss"])
    ax.set_title(plot_title)
    if i == 0:
        fig.legend((l1, l2), ("train", "val"), 'center right')
fig.text(0.5, 0.04, 'Epochs', ha='center')
fig.text(0.04, 0.5, 'Cross Entropy Loss', va='center', rotation='vertical')


# In[116]:


cnn_dev_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            cnn_results["CNN Network 1"][1][cnn_results["CNN Network 1"][3]].values,
            cnn_results["CNN Network 2"][1][cnn_results["CNN Network 2"][3]].values,
            cnn_results["CNN Network 3"][1][cnn_results["CNN Network 3"][3]].values,
            cnn_results["CNN Network 4"][1][cnn_results["CNN Network 4"][3]].values,
            cnn_results["CNN Network 5"][1][cnn_results["CNN Network 5"][3]].values,
        ],
            axis=1), 
        columns=[
            "Gen_Model", "CNN (7777 batch)", 
            "CNN (7777)", "CNN (3456)", 
            "CNN (10101010)", "CNN (7777 free)"
        ]
    )
aucs=plot_curve(
    cnn_dev_df,
    dev_Y.numpy(), 
    plot_title="Tune Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[117]:


cnn_dev_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            cnn_results["CNN Network 1"][1][cnn_results["CNN Network 1"][3]].values,
            cnn_results["CNN Network 2"][1][cnn_results["CNN Network 2"][3]].values,
            cnn_results["CNN Network 3"][1][cnn_results["CNN Network 3"][3]].values,
            cnn_results["CNN Network 4"][1][cnn_results["CNN Network 4"][3]].values,
            cnn_results["CNN Network 5"][1][cnn_results["CNN Network 5"][3]].values,
        ],
            axis=1), 
        columns=[
            "Gen_Model", "CNN (7777 batch)", 
            "CNN (7777)", "CNN (3456)", 
            "CNN (10101010)", "CNN (7777 free)"
        ]
    )
aucs=plot_curve(
    cnn_dev_df,
    dev_Y.numpy(), 
    plot_title="Tune Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[118]:


get_auc_significant_stats(
     dev_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['dev']),
    aucs
)


# In[115]:


cnn_dev_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            cnn_results["CNN Network 6"][1][cnn_results["CNN Network 6"][3]].values,
            cnn_results["CNN Network 7"][1][cnn_results["CNN Network 7"][3]].values,
            cnn_results["CNN Network 8"][1][cnn_results["CNN Network 8"][3]].values,
            cnn_results["CNN Network 9"][1][cnn_results["CNN Network 9"][3]].values,
        ],
            axis=1), 
        columns=[
            "Gen_Model","CNN (7,7,7,7) L2:0.5", 
            "CNN (7,7,7,7) L2:2", "CNN (7,7,7,7) L2:3.5", 
            "CNN (7,7,7,7) L2:5"
        ]
    )
aucs=plot_curve(
    cnn_dev_df,
    dev_Y.numpy(), 
    plot_title="Tune Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[119]:


cnn_dev_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            cnn_results["CNN Network 6"][1][cnn_results["CNN Network 6"][3]].values,
            cnn_results["CNN Network 7"][1][cnn_results["CNN Network 7"][3]].values,
            cnn_results["CNN Network 8"][1][cnn_results["CNN Network 8"][3]].values,
            cnn_results["CNN Network 9"][1][cnn_results["CNN Network 9"][3]].values,
        ],
            axis=1), 
        columns=[
            "Gen_Model","CNN (7,7,7,7) L2:0.5", 
            "CNN (7,7,7,7) L2:2", "CNN (7,7,7,7) L2:3.5", 
            "CNN (7,7,7,7) L2:5"
        ]
    )
aucs=plot_curve(
    cnn_dev_df,
    dev_Y.numpy(), 
    plot_title="Tune Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[120]:


get_auc_significant_stats(
     dev_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['dev']),
    aucs
)


# In[132]:


best_cnn_dev = cnn_results["CNN Network 7"][1][cnn_results["CNN Network 7"][3]].values 
best_lstm_dev = lstm_results["LSTM Network 12"][1][lstm_results["LSTM Network 12"][3]].values 
best_cnn_test = cnn_results["CNN Network 7"][2][cnn_results["CNN Network 7"][3]].values 
best_lstm_test = lstm_results["LSTM Network 12"][2][lstm_results["LSTM Network 12"][3]].values 


# In[133]:


final_dev_df = pd.DataFrame(
        pd.np.stack([
            gen_model_dev_df["gen_model"].values,
            best_cnn_dev,
            best_lstm_dev
        ],
            axis=1), 
        columns=["Gen_Model", "CNN (7777 l2:2)", "LSTM (250 l2:0.25)"]
    )
aucs=plot_curve(
    final_dev_df,
    dev_Y.numpy(), 
    plot_title="Tune Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[134]:


aucs=plot_curve(
    final_dev_df,
    dev_Y.numpy(), 
    plot_title="Tune Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[135]:


get_auc_significant_stats(
     dev_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['dev']),
    aucs
)


# In[154]:


final_test_df = pd.DataFrame(
        pd.np.stack([
            gen_model_test_df["gen_model"].values,
            best_cnn_test,
            best_lstm_test
        ],
            axis=1), 
        columns=["Gen_Model", "CNN", "LSTM"]
    )
aucs=plot_curve(
    final_test_df,
    test_Y.numpy(), 
    plot_title="Test Set PRC", 
    metric="PR", 
    model_type="curve"
)


# In[155]:


aucs=plot_curve(
    final_test_df,
    test_Y.numpy(), 
    plot_title="Test Set ROC", 
    metric="ROC", 
    model_type="curve"
)


# In[138]:


get_auc_significant_stats(
     test_data
    .query("sen_length < @cutoff")
    .merge(candidate_dfs['test']),
    aucs
)


# In[139]:


error_output_df = (
    candidate_dfs['dev'][[
    'candidate_id', 'disease', 
    'gene', 'doid_id', 
    'entrez_gene_id', 'sentence_id', 
    'sentence', 'curated_dsh'
    ]]
    .merge(gen_model_dev_df[['gen_model', 'candidate_id']], on="candidate_id")
    .sort_values("candidate_id")
)

error_output_df['lstm'] = best_lstm_dev
error_output_df['cnn'] = best_cnn_dev
error_output_df.head(2)


# In[152]:


error_output_df["gen_model"].hist()


# In[153]:


error_output_df["lstm"].hist()


# In[141]:


spreadsheet_name = "data/sentence_dev_error_analysis.xlsx"
writer = pd.ExcelWriter(spreadsheet_name)

(
    error_output_df
    .to_excel(writer, sheet_name='sentences', index=False)
)

if writer.engine == 'xlsxwriter':
    for sheet in writer.sheets.values():
        sheet.freeze_panes(1, 0)

writer.close()

