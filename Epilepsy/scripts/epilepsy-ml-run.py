
# coding: utf-8

# # MUST RUN AT THE START OF EVERYTHING

# In[ ]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import os
database_str = "sqlite:///" + os.environ['WORKINGPATH'] + "/Database/epilepsy.db"
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()


# # Load preprocessed data 

# To save time, this code will automatically load our labels that were generated in the previous file.

# In[ ]:

from snorkel.annotations import LabelAnnotator
labeler = LabelAnnotator(f=None)

L_train = labeler.load_matrix(session,split=0)
L_dev = labeler.load_matrix(session,split=1)
L_test = labeler.load_matrix(session,split=2)


# In[ ]:

print "Total Data Shape:"
print L_train.shape
print L_dev.shape
print L_test.shape
print

print "The number of positive candiadtes (in KB) for each division:"
print L_train[(L_train[:,0] > 0)].shape
print L_dev[(L_dev[:,0] > 0)].shape
print L_test[L_test[:,0] > 0].shape


# In[ ]:

from snorkel.annotations import FeatureAnnotator
featurizer = FeatureAnnotator()

F_train = featurizer.load_matrix(session, split=0)
F_dev = featurizer.load_matrix(session, split=1)
F_test = featurizer.load_matrix(session, split=2)


# # Run the machine learning models below

# ## Generative Model

# Since we are still in development stage below are just two generative models designed to model p(Labels,y). Until we can discuss more about the classifiers we want to use, feel free to run the below code and see some cool output.

# In[ ]:

from snorkel.learning import NaiveBayes
KB = L_train[:,0]
KB_CONTEXT = L_train
train_marginals = []
gen_model = NaiveBayes()

for models in [KB,KB_CONTEXT]:
    gen_model.train(models)
    train_marginals.append(gen_model.marginals(models))


# In[ ]:

import matplotlib.pyplot as plt
plt.hist(train_marginals[0],bins=20)
plt.title("KB")
plt.show()
plt.hist(train_marginals[1],bins=20)
plt.title("KB + Context")
plt.show()


# ## Disc Model With Hyper-Param Tuning

# In[ ]:

from snorkel.learning.utils import MentionScorer
from snorkel.learning import RandomSearch, ListParameter, RangeParameter

# Searching over learning rate
rate_param = RangeParameter('lr', 1e-6, 1e-2, step=1, log_base=10)
l1_param  = RangeParameter('l1_penalty', 1e-6, 1e-2, step=1, log_base=10)
l2_param  = RangeParameter('l2_penalty', 1e-6, 1e-2, step=1, log_base=10)


# In[ ]:

from snorkel.models import candidate_subclass
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[ ]:

from snorkel.learning import SparseLogisticRegression
import numpy as np
np.random.seed(1701)
test_marginals = []
disc_models = []
weights = []

for i,L_classes in enumerate([KB,KB_CONTEXT]):
    print i
    disc_model = SparseLogisticRegression()
    searcher = RandomSearch(session, disc_model, F_train, train_marginals[i], [rate_param, l1_param, l2_param], n=20)
    searcher.fit(F_dev, L_dev, n_epochs=50, rebalance=0.5, print_freq=25)
    disc_models.append(disc_model)
    w = disc_model.save_dict['w']
    f = w.read_value()
    values = f.eval(session = disc_model.session)
    weights.append(values)
    test_marginals.append(disc_model.marginals(F_test))


# # Generate Statistics After Model Training

# ## Grab the feature weights

# In[ ]:

from snorkel.models import FeatureKey
import numpy as np
import pandas as pd
features = session.query(FeatureKey).all()
feat_data = []
for feat, w0, w1 in zip(features,weights[0],weights[1]):
    feat_data.append([feat.name, w0[0], w1[0]])
feat_frame = pd.DataFrame(feat_data, columns= ["Feature", "Model_KB", "Model_KB_CONTEXT"])


# ## Grab the class probabilities

# In[ ]:

from snorkel.models import Candidate
import pandas as pd
test_marginals[0].shape
cand_probs = []
for candidate_id in L_test.candidate_index:
    cand = session.query(Candidate).filter(Candidate.id == candidate_id).one()
    index = L_test.candidate_index[candidate_id]
    data = [cand[0].get_span(), cand[1].get_span(),test_marginals[0][index], test_marginals[1][index]]
    data.append(cand.get_parent().text)
    data.append(L_test[:,0][index].toarray()[0][0])
    cand_probs.append(data)
cand_stats = pd.DataFrame(cand_probs, columns = ["Disease", "Gene", "Model_KB", "Model_KB_CONTEXT","Sentence","Label"])


# In[ ]:

feat_frame.sort_values("Model_KB", ascending=False, inplace=True)
feat_frame.to_csv("features.tsv", sep="\t", index=False, float_format='%.4g')
cand_stats.sort_values(['Model_KB', 'Model_KB_CONTEXT'], ascending=False, inplace=True)
cand_stats.to_csv("model_predictions.tsv",sep="\t", index=False, float_format='%.4g')


# ## Show the feature weight distribution

# In[ ]:

import seaborn as sns
sns.distplot(feat_frame["Model_KB"], kde=False)


# ## Show the class probability correlation between models

# In[ ]:

import seaborn as sns
import matplotlib.pyplot as plt
colors = sns.color_palette("muted")

# Blue is positive labels
sns.jointplot(data=cand_stats[cand_stats["Label"]==1], x="Model_KB", y="Model_KB_CONTEXT", kind="hex", color=colors[0])

#Green is the negative labels
sns.jointplot(data=cand_stats[cand_stats["Label"]==-1], x="Model_KB", y="Model_KB_CONTEXT", kind="hex", color=colors[1])


# ## Generate Curve Stats

# In[ ]:

from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc
pr_models = []
roc_models = []
for marginal in test_marginals:    
    fpr, tpr, thresholds = roc_curve(list(L_test[:,0]),marginal,pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, thresholds = precision_recall_curve(L_test[:,0].todense(),marginal,pos_label=1)
    avg_precision = average_precision_score(L_test[:,0].todense(), marginal)
    
    roc_models.append(tuple([fpr,tpr,roc_auc]))
    pr_models.append(tuple([recall,precision,avg_precision]))


# ## Precision-Recall Curves

# In[ ]:

import matplotlib.pyplot as plt
model_names = ["KB", "KB+Context"]
color_vals = ["darkorange", "cyan", "red"]
for i,model in enumerate(pr_models):
    plt.plot(model[0],model[1], color=color_vals[i],
         lw=2, label='%s (area = %0.2f)' % (model_names[i],model[2]))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower right")
plt.show()


# ## ROC-Curves

# In[ ]:

import matplotlib.pyplot as plt
model_names = ["KB", "KB+Context"]
color_vals = ["darkorange", "cyan", "red"]
for i,model in enumerate(roc_models):
    plt.plot(model[0],model[1], color=color_vals[i],
         lw=2, label='%s (area = %0.2f)' % (model_names[i],model[2]))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()


# # Parse Tree Visualization

# In[ ]:

from snorkel.models import Candidate
from snorkel.utils import get_as_dict
from tree_structs import corenlp_to_xmltree
cand = session.query(Candidate).filter(Candidate.id == 19885).one()
print cand
xmltree = corenlp_to_xmltree(get_as_dict(cand.get_parent()))
xmltree.render_tree(highlight=[range(cand[0].get_word_start(), cand[0].get_word_end() + 1), range(cand[1].get_word_start(), cand[1].get_word_end()+1)])


# In[ ]:

from treedlib import compile_relation_feature_generator
get_tdl_feats = compile_relation_feature_generator()
sids = [range(a.get_word_start(), a.get_word_end() + 1) for a in cand.get_contexts()]
tags = list(get_tdl_feats(xmltree.root, sids[0], sids[1]))
print tags

