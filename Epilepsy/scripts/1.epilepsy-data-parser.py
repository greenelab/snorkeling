
# coding: utf-8

# # MUST RUN AT THE START OF EVERYTHING

# In[ ]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

#Imports
import os
import random

from epilepsy_utils import XMLMultiDocPreprocessor
from epilepsy_utils import Tagger
import pandas as pd
from snorkel import SnorkelSession
from snorkel.candidates import PretaggedCandidateExtractor
from snorkel.models import Document, Sentence, candidate_subclass
from snorkel.parser import CorpusParser
from snorkel.viewer import SentenceNgramViewer
import tqdm


# In[ ]:

#Set up the environment
database_str = "sqlite:///" + os.environ['WORKINGPATH'] + "/Database/epilepsy.db"
os.environ['SNORKELDB'] = database_str

session = SnorkelSession()


# # Parse the Pubmed Abstracts

# The code below is designed to read and parse data gathered from pubtator. Pubtator outputs their annotated text in xml format, so that is the standard file format we are going to use. 

# In[ ]:

working_path = os.environ['WORKINGPATH']
xml_parser = XMLMultiDocPreprocessor(
    path= working_path + '/Database/epilepsy_data.xml',
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()')


# In[ ]:

working_path = os.environ['WORKINGPATH']
dg_tagger = Tagger(working_path + "/Database/epilepsy_tags_shelve")
corpus_parser = CorpusParser(fn=dg_tagger.tag)
get_ipython().magic(u'time corpus_parser.apply(list(xml_parser))')


# In[ ]:

print "Documents: ", session.query(Document).count()
print "Sentences: ", session.query(Sentence).count()


# # Get each candidate relation

# This block of code below is designed to gather and tag each sentence found. **Note**: This does include the title of each abstract.

# In[ ]:

gene_df = pd.read_csv("epilepsy-genes.tsv",sep="\t")


# In[ ]:

#This is a quick divide the documents without checking if they have gold standard or not

random.seed(100)
#Grab the sentences!!!
train_sents,dev_sents,test_sents = set(),set(),set()
docs = session.query(Document).all()
for doc in tqdm.tqdm(docs):
    for s in doc.sentences:
        in_dev = random.random() * 100 < 50
        if 'Gene' in s.entity_types:
            if ";" in s.entity_cids[s.entity_types.index('Gene')]:
                cand = s.entity_cids[s.entity_types.index('Gene')].split(";")[0]
                cand = int(cand)
            else:
                cand = int(s.entity_cids[s.entity_types.index('Gene')])
            if cand in set(gene_df[gene_df["testing"] == 1]["entrez_gene_id"]):
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

#This specifies that I want candidates that have a disease and gene mentioned in a given sentence
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[ ]:

ce = PretaggedCandidateExtractor(DiseaseGene, ['Disease', 'Gene'])


# In[ ]:

#Get the candidates from my custom tagger and then print number of candidates found
for k,sents in enumerate([train_sents, dev_sents, test_sents]):
    ce.apply(sents,split=k)
    print "Number of Candidates: ", session.query(DiseaseGene).filter(DiseaseGene.split == k).count()


# # Look at the Potential Candidates

# The one cool thing about jupyter is that you can use this tool to look at candidates. Check it out after everything above has finished running

# In[ ]:

candidates = session.query(DiseaseGene).filter(DiseaseGene.split==1)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:

sv

