
# coding: utf-8

# # Train the Discriminator for Candidate Classification on the Sentence Level

# This notebook is designed to train ML algorithms: Long Short Term Memory Neural Net (LSTM) and SparseLogisticRegression (SLR) for candidate classification. 

# ## MUST RUN AT THE START OF EVERYTHING

# Set up the database for data extraction and load the Candidate subclass for the algorithms below

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

import csv
import os

import numpy as np
import pandas as pd
import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier


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


from snorkel.annotations import FeatureAnnotator, LabelAnnotator, load_marginals
from snorkel.annotations import load_gold_labels
from snorkel.learning import SparseLogisticRegression
from snorkel.learning.disc_models.rnn import reRNN
from snorkel.learning.utils import RandomSearch
from snorkel.models import Candidate, FeatureKey, candidate_subclass


# In[ ]:


edge_type = "dg"


# In[ ]:


if edge_type == "dg":
    DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
elif edge_type == "gg":
    GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
elif edge_type == "cg":
    CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
elif edge_type == "cd":
    CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])
else:
    print("Please pick a valid edge type")


# # Load preprocessed data 

# This code will automatically load our labels and features that were generated in the [previous notebook](2.data-labeler.ipynb). 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'labeler = LabelAnnotator(lfs=[])\n\nL_train = labeler.load_matrix(session, split=0)\nL_dev = labeler.load_matrix(session, split=1)\nL_test = labeler.load_matrix(session, split=2)')


# In[ ]:


print "Total Data Shape:"
print L_train.shape
print L_dev.shape
print L_test.shape
print


# In[ ]:


get_ipython().run_cell_magic('time', '', 'featurizer = FeatureAnnotator()\n\nF_train = featurizer.load_matrix(session, split=0)\nF_dev = featurizer.load_matrix(session, split=1)\nF_test = featurizer.load_matrix(session, split=2)')


# In[ ]:


print "Total Data Shape:"
print F_train.shape
print F_dev.shape
print F_test.shape
print


# # Train Sparse Logistic Regression Disc Model

# Here we train an SLR. To find the optimal hyperparameter settings this code uses a [random search](http://scikit-learn.org/stable/modules/grid_search.html) instead of iterating over all possible combinations of parameters. After the final model has been found, it is saved in the checkpoints folder to be loaded in the [next notebook](5.data-analysis.ipynb). Furthermore, the weights for the final model are output into a text file to be analyzed as well.

# In[ ]:


L_gold_dev = load_gold_labels(session, annotator_name='danich1', split=1)
L_gold_train = load_gold_labels(session, annotator_name='danich1', split=0)


# In[ ]:


annotated_cands_train_ids = list(map(lambda x: L_gold_train.row_index[x],  L_gold_train.nonzero()[0]))
annotated_cands_dev_ids = list(map(lambda x: L_gold_dev.row_index[x],L_gold_dev.nonzero()[0]))


# In[ ]:


labeler = LabelAnnotator(lfs=[])
cids = session.query(Candidate.id).filter(Candidate.id.in_(annotated_cands_train_ids))
L_train = labeler.load_matrix(session,cids_query=cids)


# In[ ]:


words = pd.read_csv("vanilla_lstm/lstm_disease_gene_holdout/train_word_dict.csv")
train_data = pd.read_csv("vanilla_lstm/lstm_disease_gene_holdout/train_candidates_to_ids.csv")
dev_data = pd.read_csv("vanilla_lstm/lstm_disease_gene_holdout/dev_candidates_to_ids.csv")
#test_data = pd.read_csv("vanilla_lstm/lstm_disease_gene_holdout/test_candidates_to_ids.csv")

human_cur_labels = list(L_gold_train[L_gold_train != 0].toarray()[0])
sen_het_labels = np.loadtxt("vanilla_lstm/lstm_disease_gene_holdout/subsampled/train_marginals_subsampled.txt")
sen_het_labels = list(map(lambda x: 1 if x > 0.5 else -1, sen_het_labels))


# In[ ]:


vectorizer = CountVectorizer(vocabulary=words["Key"].drop_duplicates())
X = vectorizer.fit_transform(train_data.query("id in @annotated_cands_train_ids").sort_values("id")["sentence"].values)
dev_X = vectorizer.transform(dev_data.query("id in @annotated_cands_dev_ids").sort_values("id")["sentence"].values)
#test_X = vectorizer.transform(test_data["sentence"].values)


# In[ ]:


lr_model = LogisticRegression()
labels = [
    human_cur_labels,
    sen_het_labels
]

model_labels = [
    "hand_LR", 
    "all_LF_LR"
]
    
lr_grids = [
    {'C':np.linspace(1,10, num=100)} for _ in range(len(labels))
]
final_models = []


# In[ ]:


get_ipython().run_cell_magic('time', '', "for grid, y_labels in zip(lr_grids, labels):\n    fit_model = GridSearchCV(lr_model, \n                         grid, cv=10, n_jobs=3, \n                         verbose=1, scoring='roc_auc', return_train_score=True)\n    fit_model.fit(X, y_labels)\n    final_models.append(fit_model)")


# In[ ]:


print(len(final_models))


# In[ ]:


import matplotlib.pyplot as plt
for model, model_label in zip(final_models, model_labels):
    lr_result = pd.DataFrame(model.cv_results_)
    plt.plot(lr_result["param_C"], lr_result["mean_test_score"], label=model_label)
plt.legend()
plt.xlabel("C (regularization parameter)")
plt.ylabel("Mean Test Score")
plt.title("BOW Training CV (10-fold)")


# In[ ]:


for i, label in zip(range(len(final_models)), model_labels):
    lr_weights = pd.DataFrame(list(zip(final_models[i].best_estimator_.coef_[0], vectorizer.get_feature_names())), columns=["Weight", "Feature"])
    print(label)
    print(lr_weights.sort_values("Weight", ascending=False).head(10))
    print()


# In[ ]:


lr_marginals = []
for model in final_models:
    lr_marginals.append(model.best_estimator_.predict_proba(dev_X)[:,1])


# In[ ]:


print(pd.Series(final_models[0].best_estimator_.predict(dev_X)).value_counts())
print()
print(pd.Series(final_models[1].best_estimator_.predict(dev_X)).value_counts())
print()


# In[ ]:


for marginal, model_label in zip(lr_marginals,model_labels):
    filename = "vanilla_lstm/lstm_disease_gene_holdout/subsampled/lf_marginals/{}_dev_marginals.csv".format(model_label)
    pd.DataFrame(marginal,
             columns=["LR_Marginals"]
        ).to_csv(filename, index=False)


# # DO NOT RUN  BELOW

# ## Train a LSTM Disc Model

# This block of code trains an LSTM. An LSTM is a special type of recurrent nerual network that retains a memory of past values over period of time. ([Further explaination here](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)). The problem with the code below is that sqlalchemy runs into an out of memory error on my computer during the preprocessing step. As a consequence we have to resort loading this data onto University of Pennsylvania's Performance Computing Cluster. The data that gets preprocessed is exported to a text file and then get shipped towards the cluster.

# In[ ]:


directory = 'vanilla_lstm/lstm_disease_gene_holdout/subsampled'


# In[ ]:


get_ipython().run_line_magic('time', 'train_marginals = load_marginals(session, split=0)')
np.savetxt("{}/train_marginals".format(directory), train_marginals)


# In[ ]:


def read_word_dict(filename):
    """
     Read a CSV into a dictionary using the Key column (as string) and Value column (as int).
    Keywords:
    fielname - name of the file to read
    """
    data = {}
    with open(filename, 'r') as f:
        input_file = csv.DictReader(f)
        for row in tqdm.tqdm(input_file):
            data[row['Key']] = int(row['Value'])
    return data


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from snorkel.learning.disc_models.rnn.utils import SymbolTable\n\n"""\ntrain_kwargs = {\n    \'lr\':         0.001,\n    \'dim\':        100,\n    \'n_epochs\':   10,\n    \'dropout\':    0.5,\n    \'print_freq\': 1,\n    \'max_sentence_length\': 1000,\n}\n"""\nword_dict = read_word_dict("vanilla_lstm/lstm_disease_gene_holdout/train_word_dict.csv")\nlstm = reRNN(seed=100, n_threads=4)\nlstm.word_dict = SymbolTable()\nlstm.word_dict.d = word_dict\n#lstm.train(train_cands, train_marginals[0:10], X_dev=dev_cands, Y_dev=L_dev[0:10], **train_kwargs)')


# ### Write the Training data to an External File

# In[ ]:


get_ipython().run_cell_magic('time', '', 'field_names = [\n    "disease_id", "disease_name",\n    "disease_char_start", "disease_char_end",\n    "gene_id", "gene_name",\n    "gene_char_start", "gene_char_end",\n    "sentence", "pubmed"\n]\nchunksize = 100000\nstart = 0\n\nwith open(\'{}/train_candidates_ends.csv\'.format(directory), \'w\') as g:\n    with open("{}/train_candidates_offsets.csv".format(directory), "w") as f:\n        with open("{}/train_candidates_sentences.csv".format(directory), "w") as h:\n            output = csv.writer(f)\n            writer = csv.DictWriter(h, fieldnames=field_names)\n            writer.writeheader()\n\n            while True:\n                train_cands = (\n                        session\n                        .query(DiseaseGene)\n                        .filter(DiseaseGene.split == 0)\n                        .order_by(DiseaseGene.id)\n                        .limit(chunksize)\n                        .offset(start)\n                        .all()\n                )\n\n                if not train_cands:\n                    break\n\n                for c in tqdm.tqdm(train_cands):\n                    data, ends = lstm._preprocess_data([c], extend=True)\n                    output.writerow(data[0])\n                    g.write("{}\\n".format(ends[0]))\n\n                    row = {\n                    "disease_id": c.Disease_cid,"disease_name":c[0].get_span(),\n                    "disease_char_start":c[0].char_start, "disease_char_end": c[0].char_end, \n                    "gene_id": c.Gene_cid, "gene_name":c[1].get_span(), \n                    "gene_char_start":c[1].char_start, "gene_char_end":c[1].char_end, \n                    "sentence": c.get_parent().text, "pubmed": c.get_parent().get_parent().name\n                    }\n\n                    writer.writerow(row)\n\n                start += chunksize')


# ### Save the word dictionary to an External File

# In[ ]:


get_ipython().run_cell_magic('time', '', 'with open("{}/train_word_dict.csv".format(directory), \'w\') as f:\n    output = csv.DictWriter(f, fieldnames=["Key", "Value"])\n    output.writeheader()\n    for key in tqdm.tqdm(lstm.word_dict.d):\n        output.writerow({\'Key\':key, \'Value\': lstm.word_dict.d[key]})')


# ### Save the Development Candidates to an External File

# In[ ]:


dev_cands = (
    session
    .query(DiseaseGene)
    .filter(DiseaseGene.id.in_(list(map(int,train_data.sort_values("id")["id"].values))))
    .all()
)
dev_cand_labels = pd.read_csv("stratified_data/train_set.csv")
hetnet_set = set(map(tuple,dev_cand_labels[dev_cand_labels["hetnet"] == 1][["disease_id", "gene_id"]].values))


# In[ ]:


dev_cands = (
    session
    .query(DiseaseGene)
    .filter(DiseaseGene.id.in_(annotated_cands))
    .all()
)
dev_cand_labels = pd.read_csv("stratified_data/dev_set.csv")
hetnet_set = set(map(tuple,dev_cand_labels[dev_cand_labels["hetnet"] == 1][["disease_id", "gene_id"]].values))


# In[ ]:


dev_cands = (
        session
        .query(DiseaseGene)
        .filter(DiseaseGene.split == 1)
        .order_by(DiseaseGene.id)
        .all()
)

dev_cand_labels = pd.read_csv("stratified_data/dev_set.csv")
hetnet_set = set(map(tuple,dev_cand_labels[dev_cand_labels["hetnet"] == 1][["disease_id", "gene_id"]].values))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'field_names = [\n    "disease_id", "disease_name",\n    "disease_char_start", "disease_char_end",\n    "gene_id", "gene_name",\n    "gene_char_start", "gene_char_end",\n    "sentence", "pubmed"\n]\n\nwith open(\'{}/delete_candidates_offset.csv\'.format(directory), \'w\') as g:\n    with open(\'{}/train_candidates_ends.csv\'.format(directory), \'w\') as f:\n        with open(\'{}/delete_candidates_sentences.csv\'.format(directory), \'w\') as h:\n            \n            output = csv.writer(g)\n            #label_output = csv.writer(f)\n            writer = csv.DictWriter(h, fieldnames=field_names)\n            writer.writeheader()\n            \n            for c in tqdm.tqdm(dev_cands):\n                data, ends = lstm._preprocess_data([c])\n                output.writerow(data[0])\n                f.write("{}\\n".format(ends[0]))\n                #label_output.writerow([1 if (c.Disease_cid, int(c.Gene_cid)) in hetnet_set else -1])\n\n                row = {\n                "disease_id": c.Disease_cid,"disease_name":c[0].get_span(),\n                "disease_char_start":c[0].char_start, "disease_char_end": c[0].char_end, \n                "gene_id": c.Gene_cid, "gene_name":c[1].get_span(), \n                "gene_char_start":c[1].char_start, "gene_char_end":c[1].char_end, \n                "sentence": c.get_parent().text, "pubmed": c.get_parent().get_parent().name\n                }\n\n                writer.writerow(row) ')


# ### Save the Test Candidates to an External File

# In[ ]:


test_cands = (
        session
        .query(DiseaseGene)
        .filter(DiseaseGene.split == 2)
        .order_by(DiseaseGene.id)
        .all()
)

dev_cand_labels = pd.read_csv("stratified_data/test_set.csv")
hetnet_set = set(map(tuple,dev_cand_labels[dev_cand_labels["hetnet"] == 1][["disease_id", "gene_id"]].values))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'field_names = [\n    "disease_id", "disease_name",\n    "disease_char_start", "disease_char_end",\n    "gene_id", "gene_name",\n    "gene_char_start", "gene_char_end",\n    "sentence", "pubmed"\n]\n\nwith open(\'{}/test_candidates_offset.csv\'.format(directory), \'w\') as g:\n    with open(\'{}/test_candidates_labels.csv\'.format(directory), \'w\') as f:\n        with open(\'{}/test_candidates_sentences.csv\'.format(directory), \'w\') as h:\n            \n            output = csv.writer(g)\n            label_output = csv.writer(f)\n            writer = csv.DictWriter(h, fieldnames=field_names)\n            writer.writeheader()\n            \n            for c in tqdm.tqdm(test_cands):\n                data, ends = lstm._preprocess_data([c])\n                output.writerow(data[0])\n                label_output.writerow([1 if (c.Disease_cid, int(c.Gene_cid)) in hetnet_set else -1])\n\n                row = {\n               "disease_id": c.Disease_cid,"disease_name":c[0].get_span(),\n                "disease_char_start":c[0].char_start, "disease_char_end": c[0].char_end, \n                "gene_id": c.Gene_cid, "gene_name":c[1].get_span(), \n                "gene_char_start":c[1].char_start, "gene_char_end":c[1].char_end, \n                "sentence": c.get_parent().text, "pubmed": c.get_parent().get_parent().name\n                }\n\n                writer.writerow(row) ')

