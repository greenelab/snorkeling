import os

import matplotlib.pyplot as plt
from snorkel import SnorkelSession
from snorkel.annotations import LabelAnnotator
from snorkel.learning import NaiveBayes

os.environ['SNORKELDB'] = 'sqlite:///snorkel.db'
session = SnorkelSession()

# set the constants
TRAIN = 0
DEV = 1

# Grab the labels from the previous script
labeler = LabelAnnotator(f=None)
L_train = labeler.load_matrix(session, split=TRAIN)
L_dev = labeler.load_matrix(session, split=DEV)

# Train the models
gen_model = NaiveBayes()
gen_model.train(L_train)
train_marginals = gen_model.marginals(L_train)

plt.figure()
plt.hist(train_marginals, bins=20)
plt.savefig('results.png')
