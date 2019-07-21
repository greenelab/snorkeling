
# coding: utf-8

# # CoCoScore Implementation

# This notebook consists of implementing the [CoCoScore](https://www.biorxiv.org/content/10.1101/444398v1) literature model for comparison.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import defaultdict
import os
import pickle
import re
import sys

sys.path.append(os.path.abspath('../../../../modules'))

import operator
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, accuracy_score, confusion_matrix
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
from utils.notebook_utils.dataframe_helper import load_candidate_dataframes, generate_results_df
from utils.notebook_utils.plot_helper import plot_curve, plot_label_matrix_heatmap


# In[4]:


CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])


# In[5]:


quick_load = True


# In[6]:


total_candidates_df = pd.read_table("../../dataset_statistics/results/all_ctd_map.tsv.xz")
total_candidates_df.head(2)


# In[7]:


spreadsheet_names = {
    'dev': '../../data/sentences/sentence_labels_dev.xlsx',
    'test': '../../data/sentences/sentence_labels_test.xlsx'
}


# In[8]:


candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key], "curated_ctd")
    for key in spreadsheet_names
}

for key in candidate_dfs:
    print("Size of {} set: {}".format(key, candidate_dfs[key].shape[0]))


# In[9]:


distant_supervision_marginals = pd.read_table("../../label_sampling_experiment/results/CtD/marginals/baseline_marginals.tsv.xz")
distant_supervision_marginals.head(2)


# In[10]:


all_embedded_cd_df = pd.read_table("../../word_vector_experiment/results/all_embedded_cd_sentences.tsv.xz")
all_embedded_cd_df.head(2)


# In[11]:


word_vectors = pd.read_csv(
    "../../word_vector_experiment/results/compound_treats_disease_word_vectors.bin",
    sep=" ", skiprows=1, 
    header=None, index_col=0, 
    keep_default_na=False
)
word_vectors.head(2)


# In[12]:


word_dict = pd.read_table("../../word_vector_experiment/results/compound_treats_disease_word_dict.tsv", keep_default_na=False)
reverse_word_dict = dict(zip(word_dict.index, word_dict.word))
word_dict = dict(zip(word_dict.word, word_dict.index))


# In[13]:


total_training_sentences_df = (
    all_embedded_cd_df.merge(
        distant_supervision_marginals
        .assign(labels=lambda x: x.pos_class_marginals > 0.5)
        [["labels", "candidate_id"]]
        .astype({"labels":int}),
        on="candidate_id"
    )
)
total_training_sentences_df.head(2)


# In[14]:


total_dev_sentences_df = (
    all_embedded_cd_df.merge(
        candidate_dfs['dev']
        [["curated_ctd", "candidate_id"]],
        on="candidate_id"
    )
)
total_dev_sentences_df.head(2)


# In[15]:


total_test_sentences_df = (
    all_embedded_cd_df.merge(
        candidate_dfs['test']
        [["curated_ctd", "candidate_id"]],
        on="candidate_id"
    )
)
total_test_sentences_df.head(2)


# In[16]:


def create_data_matrix(query_df, filename="sentences.txt"):
    search_regex = rf'(\b{word_dict["~~[[2"]}\b.+\b{word_dict["2]]~~"]}\b,)'
    search_regex += rf'|(\b{word_dict["~~[[1"]}\b.+\b{word_dict["1]]~~"]}\b,)'
    
    print(search_regex)
    data = []
    with open(filename, "w") as g:
        for index, row in tqdm_notebook(query_df.iterrows()):
            cand_str = ",".join(map(str, row.dropna().astype(int).values))
            pruned_str = re.sub(search_regex, "", cand_str)
            values = list(map(int, pruned_str.split(",")))
            g.write(f"__label__{values[-1]}\t")
            g.write("\t".join([reverse_word_dict[val] for val in values[:-2]]))
            g.write("\n")


# In[17]:


query_df = (
    total_training_sentences_df
    [[col for col in total_training_sentences_df.columns if col not in ["sen_length"]]]
)
create_data_matrix(query_df, filename="training.txt")


# In[18]:


query_df = (
    total_dev_sentences_df
    [[col for col in total_dev_sentences_df.columns if col not in ["sen_length"]]]
)
create_data_matrix(query_df, filename="dev.txt")


# In[19]:


query_df = (
    total_test_sentences_df
    [[col for col in total_test_sentences_df.columns if col not in ["sen_length"]]]
)
create_data_matrix(query_df, filename="test.txt")


# In[24]:


os.system("../../../../../fastText/fasttext supervised -input training.txt -output ctd_model -lr 0.005 -epoch 50 -dim 300 -wordNgrams 2")
os.system("../../../../../fastText/fasttext predict-prob ctd_model.bin dev.txt > dev_predictions.tsv")
os.system("../../../../../fastText/fasttext predict-prob ctd_model.bin test.txt > test_predictions.tsv")


# In[25]:


precision, recall, _ = precision_recall_curve(
    total_dev_sentences_df.curated_ctd,
    pd.read_table('dev_predictions.tsv', header=None, sep=" ")[1]
)
auc(recall, precision)


# In[26]:


plt.plot(recall, precision)


# In[27]:


fpr, tpr, _ = roc_curve(
    total_dev_sentences_df.curated_ctd,
    pd.read_table('dev_predictions.tsv', header=None, sep=" ")[1]
)
auc(fpr, tpr)


# In[28]:


plt.plot(fpr, tpr)


# In[31]:


query_df = (
    all_embedded_cd_df
    .assign(labels=0)
    [[col for col in all_embedded_cd_df.columns if col not in ["sen_length"]]]
)
create_data_matrix(query_df, filename="all_cd_sentences.txt")


# In[32]:


os.system("../../../../../fastText/fasttext predict-prob ctd_model.bin all_cd_sentences.txt > all_cd_sentences_predictions.tsv")


# In[34]:


predictions_df = pd.read_table("all_cd_sentences_predictions.tsv", header=None, names=["label", "predictions"], sep=" ")
predictions_df['candidate_id'] = all_embedded_cd_df.candidate_id.values
predictions_df.head(2)


# In[35]:


final_pred_df = (
    total_candidates_df
    [["doid_id", "drugbank_id", "candidate_id"]]
    .merge(predictions_df[["predictions", "candidate_id"]])
)
final_pred_df.head(2)


# In[36]:


added_scores_df = (
    final_pred_df
    .groupby(["doid_id", "drugbank_id"])
    .aggregate({"predictions": 'sum'})
    .reset_index()
)
added_scores_df.head(2)


# In[38]:


total_score = added_scores_df.predictions.sum()
disease_scores = added_scores_df.groupby("doid_id").agg({"predictions":"sum"}).reset_index()
disease_scores = dict(zip(disease_scores.doid_id, disease_scores.predictions))
drug_scores = added_scores_df.groupby("drugbank_id").agg({"predictions":"sum"}).reset_index()
drug_scores = dict(zip(drug_scores.drugbank_id, drug_scores.predictions))

alpha=0.65

final_scores_df = added_scores_df.assign(
    final_score=(
        added_scores_df.apply(
            lambda x: pd.np.exp(
                    alpha*pd.np.log(x['predictions']) + (1-alpha)*(
                    pd.np.log(x['predictions']) + pd.np.log(total_score) - 
                    pd.np.log(disease_scores[x['doid_id']]) - pd.np.log(drug_scores[x['drugbank_id']])
                )
            ), 
            axis=1
        )
    )
)
final_scores_df.head(2)


# In[39]:


score_with_labels_df = (
    final_scores_df
    .merge(
        total_candidates_df[["drugbank_id", "doid_id", "hetionet"]],
        on=["drugbank_id", "doid_id"]
    )
    .drop_duplicates()
)
score_with_labels_df.head(2)


# In[41]:


score_with_labels_df.drop("predictions", axis=1).to_csv("cd_edge_prediction_cocoscore.tsv", sep="\t", index=False)


# In[40]:


fpr, tpr, _ = roc_curve(score_with_labels_df.hetionet, score_with_labels_df.final_score)
print(auc(fpr, tpr))

precision, recall, _ = precision_recall_curve(score_with_labels_df.hetionet, score_with_labels_df.final_score)
print(auc(recall, precision))

