
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


# # Parse the Pubmed Abstracts

# The code below is designed to read and parse data gathered from pubtator. Pubtator outputs their annotated text in xml format, so that is the standard file format we are going to use. 

# In[ ]:

from epilepsy_utils import XMLMultiDocPreprocessor
import os
working_path = os.environ['WORKINGPATH']
xml_parser = XMLMultiDocPreprocessor(
    path= working_path + '/Database/epilepsy_data.xml',
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()')


# In[ ]:

from epilepsy_utils import Tagger
from snorkel.parser import CorpusParser
import os
working_path = os.environ['WORKINGPATH']
dg_tagger = Tagger(working_path + "/Database/epilepsy_tags_shelve")
corpus_parser = CorpusParser(fn=dg_tagger.tag)
get_ipython().magic(u'time corpus_parser.apply(list(xml_parser))')


# In[ ]:

from snorkel.models import Document, Sentence

print "Documents: ", session.query(Document).count()
print "Sentences: ", session.query(Sentence).count()


# # Get each candidate relation

# This block of code below is designed to gather and tag each sentence found. **Note**: This does include the title of each abstract.

# In[ ]:

import pandas as pd
gene_list = pd.read_csv("epilepsy-genes.tsv",sep="\t")


# In[ ]:

#This is a quick divide the documents without checking if they have gold standard or not
from snorkel.models import Document
import tqdm
import random

random.seed(100)
#Grab the sentences!!!
train_sents,dev_sents,test_sents = set(),set(),set()
docs = session.query(Document).all()
for doc in tqdm.tqdm(docs):
    for s in doc.sentences:
        in_dev = True if random.random() * 100 < 50 else False
        if 'Gene' in s.entity_types:
            if ";" in s.entity_cids[s.entity_types.index('Gene')]:
                cand = s.entity_cids[s.entity_types.index('Gene')].split(";")[0]
                cand = int(cand)
            else:
                cand = int(s.entity_cids[s.entity_types.index('Gene')])
            if cand in set(gene_list[gene_list["testing"] == 1]["entrez_gene_id"]):
                test_sents.add(s)
            else:
                if in_dev:
                    dev_sents.add(s)
                else:
                    train_sents.add(s)
        else:
            if in_dev:
                dev_sents.add(s)
            else:
                train_sents.add(s)


# In[ ]:

print len(train_sents)
print len(dev_sents)
print len(test_sents)


# In[ ]:

from snorkel.models import candidate_subclass

#This specifies that I want candidates that have a disease and gene mentioned in a given sentence
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[ ]:

from snorkel.candidates import PretaggedCandidateExtractor

ce = PretaggedCandidateExtractor(DiseaseGene, ['Disease', 'Gene'])


# In[ ]:

#Get the candidates from my custom tagger and then print number of candidates found
for k,sents in enumerate([train_sents,dev_sents,test_sents]):
    ce.apply(sents,split=k)
    print "Number of Candidates: ", session.query(DiseaseGene).filter(DiseaseGene.split == k).count()


# # Look at the Potential Candidates

# The one cool thing about jupyter is that you can use this tool to look at candidates. Check it out after everything above has finished running

# In[ ]:

from snorkel.viewer import SentenceNgramViewer

candidates = session.query(DiseaseGene).filter(DiseaseGene.split==1)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:

sv

