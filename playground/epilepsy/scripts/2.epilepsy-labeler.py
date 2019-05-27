
# coding: utf-8

# # MUST RUN AT THE START OF EVERYTHING

# In[ ]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

import re
import os

from snorkel import SnorkelSession
from snorkel.annotations import FeatureAnnotator
from snorkel.annotations import LabelAnnotator
from snorkel.models import candidate_subclass
from snorkel.viewer import SentenceNgramViewer
from snorkel.lf_helpers import (
    get_left_tokens,
    get_right_tokens, 
    get_between_tokens,
    get_tagged_text,
    get_text_between,
    rule_regex_search_tagged_text,
    rule_regex_search_btw_AB,
    rule_regex_search_btw_BA,
    rule_regex_search_before_A,
    rule_regex_search_before_B,
)
import pandas as pd


# In[ ]:

database_str = "sqlite:///" + os.environ['WORKINGPATH'] + "/Database/epilepsy.db"
os.environ['SNORKELDB'] = database_str


session = SnorkelSession()


# In[ ]:

DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# # Look at potential Candidates

# Use this to look at loaded candidates from a given set. The constants represent the index to retrieve the training set, development set and testing set.

# In[ ]:

TRAIN = 0
DEV = 1
TEST = 2

candidates = session.query(DiseaseGene).filter(DiseaseGene.split==TRAIN).all()
sv = SentenceNgramViewer(candidates, session)


# In[ ]:

sv


# # Label Functions

# Here is the fundamental part of the project. Below are the label functions that are used to give a candidate a label of 1,0 or -1 which corresponds to correct relation, not sure and incorrection relation. The goal here is to develop functions that can label as many candidates as possible.

# In[ ]:

gene_list = pd.read_csv('epilepsy-genes.tsv',sep="\t")


# In[ ]:

variation_words = {"mutation", "mutation-negative", "mutation-positive",
                   "de novo", "heterozygous", "homozygous", "deletion", "variants","variant",
                   "recessive","autosomal", "haploinsufficiency", "knock-out", "genotype",
                  "null"}

cause_words = {"cause", "caused", "maps", "due", "associated"}
neg_words = {"serum", "intervention", "negative"}

related_diseases_symptoms = {"epileptic encephalopathies","epileptic encephalopathy", "seizures", "encephalopathy",
                             "epileptic spasms", "myoclonic astatic epilepsy", "neurodevelopmental",
                            "refractory epilepsy", "severe myoclonic epilepsy of infancy",
                            "dravet syndrome", "myoclonic-astatic epilepsy", "absence epilepsy", 
                            "epilepsies", "west syndrome", "seizures", "autoimmune epilepsy",
                            "temporal libe epilepsy"}

unrelated_diseases = {"continguous gene syndrome", "X-linked clinical syndrome", 
                      "insulin-dependent diabetes mellitus", "stiff-person syndrome",
                     "vascular syndrome","autophagic vacuolar myopathy","cardiomyopathy",
                     "Chinese linear nevus sebaceous syndrome", "MPPH syndrome",
                     "hypertrophic cardiomyopathy", "mowat-wilson syndrome", "hunter syndrome",
                     "Nephrotic Syndrome", "Vici syndrome"}

disese_abbreviations_pos = {"SMEI", "DS", "MAE", "CS", "TLE", "ADLTE", "EIEE"}
disease_abbreviations_neg = {"MWS", "KDT", "TSC", "CCHS", "IDDM", "BPP", "NS"}

gene_adj = {"-related", "anti-", "-gene"}
model_organisms = {"mice", "zebrafish", "drosophila"}

disease_context = {"patients with", "individuals with", "cases of", "treatment of"}

def LF_abbreviation(c):
    """
    IF {{B}}} {{A}} or vice versa then not a valid relationship
    """
    if len(get_text_between(c)) < 3:
        return -1
    return 0

def LF_is_a(c):
    """
    If {{a}} is a {{B}} or {{B}} is a {{A}}
    """
    return rule_regex_search_btw_AB(c,r'.* is a .*',-1) or rule_regex_search_btw_BA(c,r'is a',-1)

def LF_variation(c):
    """
    If variation keyword in close proximity then label as positive
    """
    if len(variation_words.intersection(get_left_tokens(c[1]))) > 0:
        return 1
    if len(variation_words.intersection(get_right_tokens(c[1]))) > 0:
        return 1
    return 0

def LF_model_organisms(c):
    """
    If mentions model organism then c[1] should be a gene
    """
    if len(model_organisms.intersection(get_left_tokens(c[1]))) > 0:
        return 1
    if len(model_organisms.intersection(get_left_tokens(c[1]))) > 0:
        return 1
    return 0

def LF_cause(c):
    """
    If the causual keywords are between disease and gene then should be positive predictor
    """
    if len(cause_words.intersection(get_between_tokens(c))) > 0:
        return 1
    return 0

def LF_neg_words(c):
    """
    If it mentions serum or intervention before or after gene then negative 
    """
    if len(neg_words.intersection(get_left_tokens(c[1],window=3))) > 0:
        return -1
    if len(neg_words.intersection(get_right_tokens(c[1],window=3))) > 0:
        return -1
    return 0

def LF_gene(c):
    """
    If candidate has gene word near it
    """
    if "gene" in get_left_tokens(c[1]) or "gene" in get_right_tokens(c[1]):
        return 1
    return 0

def LF_symptoms(c):
    """
    Add epilepsy specific symptoms
    """
    if c[0].get_span().lower() in related_diseases_symptoms:
        return 1
    return -1

def LF_disease_abbreviations(c):
    """
    Label abbreviations
    """
    if c[0].get_span().lower() in disease_abbreviations_pos:
        return 1
    if c[0].get_span().lower() in disease_abbreviations_neg:
        return -1
    return 0
 
def LF_unrelated_disease(c):
    """
    If the disease is completely unrelated remove
    """
    if c[0].get_span() in unrelated_diseases:
        return -1
    return 0

def LF_related_adj(c):
    """
    If there is a GENE with a -related tag next to it, then it might be important.
    """
    for adj in gene_adj:
        if adj in c[1].get_span().lower():
            return 1
    return 0

def LF_disease_context(c):
    """
    If mentions cases of or patients with -> disease
    """
    tokens = "".join(get_left_tokens(c[1],window=3))
    for context in disease_context:
        if context in tokens:
            return -1
    return 0


# # Distant Supervision

# In[ ]:

def LF_KB(c):
    """
    If in knowledge base
    """
    if c[0].sentence.entity_cids[c[0].get_word_start()] == "D004827":
        if ";" in c[1].sentence.entity_cids[c[1].get_word_start()]:
            gene_id = int(c[1].sentence.entity_cids[c[1].get_word_start()].split(";")[0])
        else:
            gene_id = int(c[1].sentence.entity_cids[c[1].get_word_start()])
        if gene_id in set(gene_list[gene_list["positive"]==1]["entrez_gene_id"]):
            return 1
    return -1

def LF_is_gene(c):
    """
    If the name is a gene
    """
    if c[1].get_span() in set(gene_list["gene_name"]) or c[1].get_span() in set(gene_list["gene_symbol"]):
        return 0
    return -1


# # Debug Label Function

# In[ ]:

def LF_DEBUG(C):
    print "Left Tokens"
    print get_left_tokens(c,window=3)
    print
    print "Right Tokens"
    print get_right_tokens(c)
    print
    print "Between Tokens"
    print get_between_tokens(c)
    print 
    print "Tagged Text"
    print get_tagged_text(c)
    print re.search(r'{{B}} .* is a .* {{A}}',get_tagged_text(c))
    print
    print "Get between Text"
    print get_text_between(c)
    print len(get_text_between(c))
    print 
    print "Parent Text"
    print c.get_parent()
    print
    return 0


# In[ ]:

LFs = [
    #Distant Supervision
    LF_KB, LF_is_gene,
    
    #Other Label Functions
    LF_abbreviation,LF_is_a,
    LF_variation,LF_cause,LF_neg_words,
    LF_gene, LF_symptoms, LF_is_gene,
    LF_model_organisms, LF_unrelated_disease,
    LF_related_adj, LF_disease_context
]


# # Test out Label Functions

# In[ ]:

labeled = []
candidates = session.query(DiseaseGene).filter(DiseaseGene.split == 0).all()
#candidates = [session.query(DiseaseGene).filter(DiseaseGene.id == ids).one() for ids in [19817,19818,19830,19862,19980,20001,20004]]

for c in candidates:
    if c[0].get_parent().id != 14264:
        continue
    print c
    print get_tagged_text(c)
    print c[1].sentence.entity_cids[c[1].get_word_start()]


# # Label The Candidates

# This block of code will run through the label functions and label each candidate in the training and development groups.

# In[ ]:

labeler = LabelAnnotator(f=LFs)

get_ipython().magic(u'time L_train = labeler.apply(split=0)')
get_ipython().magic(u'time L_dev = labeler.apply_existing(split=1)')
get_ipython().magic(u'time L_test = labeler.apply_existing(split=2)')


# In[ ]:

featurizer = FeatureAnnotator()

get_ipython().magic(u'time F_train = featurizer.apply(split=0)')
get_ipython().magic(u'time F_dev = featurizer.apply_existing(split=1)')
get_ipython().magic(u'time F_test = featurizer.apply_existing(split=2)')


# # Generate Coverage Stats

# Before throwing our labels at a machine learning algorithm take a look at some quick stats. The code below will show the coverage of each label function and some other stat things. 

# In[ ]:

print L_train.lf_stats(session, )


# In[ ]:

print L_train.get_candidate(session,21)
print L_train.get_candidate(session,21).get_parent()


# In[ ]:

print L_train.shape
print L_train[L_train < 0].shape
print L_train[:,0]


# In[ ]:

print L_dev.lf_stats(session, )

