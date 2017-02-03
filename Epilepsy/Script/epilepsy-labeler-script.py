import os
import re

# Label functions come from seperate script
from labelers import *

from snorkel import SnorkelSession
from snorkel.annotations import LabelAnnotator
from snorkel.models import candidate_subclass, Document

# Set the constants
os.environ['SNORKELDB'] = 'sqlite:///snorkel.db'
TRAIN = 0
DEV = 1

session = SnorkelSession()
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])

LFs = [
    LF_between_tag,
    LF_mutation,
    LF_check_disease_tag,
]
print "Documents: {}".format(session.query(Document).count())
labeler = LabelAnnotator(f=LFs)
L_train = labeler.apply(split=TRAIN)
L_dev = labeler.apply_existing(split=DEV)

print L_train.lf_stats(session, )
print L_dev.lf_stats(session, )
