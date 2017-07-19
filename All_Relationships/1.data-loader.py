
# coding: utf-8

# # MUST RUN AT THE START OF EVERYTHING

# In[1]:


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

#Imports
import csv
import os
import random

import numpy as np
import pandas as pd
import tqdm


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


from snorkel.candidates import PretaggedCandidateExtractor
from snorkel.models import Document, Sentence, candidate_subclass
from snorkel.parser import CorpusParser
from snorkel.viewer import SentenceNgramViewer
from utils.bigdata_utils import XMLMultiDocPreprocessor
from utils.bigdata_utils import Tagger
from sqlalchemy import func


# # Parse the Pubmed Abstracts

# The code below is designed to read and parse data gathered from pubtator. Pubtator outputs their annotated text in xml format, so that is the standard file format we are going to use. 

# In[ ]:


get_ipython().magic(u"time filter_df = pd.read_table('https://github.com/greenelab/pubtator/raw/631e86002e11c41cfcfb0043e60b84ab321bdae3/data/pubtator-hetnet-tags.tsv.xz')")


# In[ ]:


get_ipython().magic(u"time grouped = filter_df.groupby('pubmed_id')")


# In[ ]:


# Please change to your local document here
working_path = '/home/danich1/Documents/Database/pubmed_docs.xml'
xml_parser = XMLMultiDocPreprocessor(
    path= working_path,
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()', tag_filter=set(filter_df['pubmed_id']))


# In[ ]:


dg_tagger = Tagger(grouped)


# In[ ]:


corpus_parser = CorpusParser(fn=dg_tagger.tag)
document_chunk = []

for document in tqdm.tqdm(xml_parser.generate()):
    
    document_chunk.append(document)

    # chunk the data because snorkel cannot 
    # scale properly yet
    if len(document_chunk) >= 5e4:
        corpus_parser.apply(document_chunk, parallelism=5, clear=False)
        document_chunk = []
    
# If generator exhausts and there are still
# document to parse
if len(document_chunk) > 0:
    corpus_parser.apply(data, parallelism=5, clear=False)
    document_chunk = []


# # Get each candidate relation

# This block of code below is designed to gather and tag each sentence found. **Note**: This does include the title of each abstract.

# In[4]:


chunk_size = 2e5


# In[ ]:


def insert_cand_to_db(extractor, sentences):
    for split, sens in enumerate(sentences):
        extractor.apply(sens, split=split, parallelism=5, clear=False)


# In[ ]:


def print_candidates(context_class, edge):
    for i, label in enumerate(["Train", "Dev", "Test"]):
        cand_len = session.query(context_class).filter(context_class.split == i).count()
        print("Number of Candidates for {} edge and {} set: {}".format(edge, label, cand_len))


# In[ ]:


#This specifies the type of candidates to extract
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
dge = PretaggedCandidateExtractor(DiseaseGene, ['Disease', 'Gene'])

GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
gge = PretaggedCandidateExtractor(GeneGene, ['Gene', 'Gene'])

CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
cge = PretaggedCandidateExtractor(CompoundGene, ['Compound', 'Gene'])

CompoundDisease = candidate_subclass('CompoundDisease', ['Compound','Disease'])
cde = PretaggedCandidateExtractor(CompoundDisease, ['Compound', 'Disease'])


# In[ ]:


# set the seed for reproduction
np.random.seed(100)
total_sentences = 61615305


# In[ ]:


category_list = np.random.choice([0,1,2], total_sentences, p=[0.7,0.2,0.1])


# In[ ]:


# Divide the sentences into train, dev and test sets
   
#Grab the sentences!!!
train_sens = set()
dev_sens = set()
test_sens = set()

offset = 0
category_index = 0
sql_query = session.query(Document).limit(chunk_size)

#divde and insert into the database
while True:
    documents = list(sql_query.offset(offset).all())
    
    if not documents:
        break
        
    for doc in tqdm.tqdm(documents): 
        for s in doc.sentences:
            
            # Stratify the data into train, dev, test 
            category = category_list[category_index]
            set_index = set_index + 1
            
            if category == 0:
                train_sens.add(s)
            elif category == 1:
                dev_sens.add(s)
            else:
                test_sens.add(s)

    # insert all the edge types
    for edges in [dge, gge, cge, cde]:
        insert_cand_to_db(edges, [train_sens, dev_sens, test_sens])
        
    offset = offset + chunk_size

    #Reset for each chunk
    train_sens = set()
    dev_sens = set()
    test_sens = set()


# In[ ]:


print_candidates(DiseaseGene, 'DiseaseGene')
print_candidates(GeneGene, 'GeneGene')
print_candidates(CompoundGene, 'CompoundGene')
print_candidates(CompoundDisease, 'CompoundDisease')


# # Look at the Potential Candidates

# The one cool thing about jupyter is that you can use this tool to look at candidates. Check it out after everything above has finished running

# In[ ]:


TRAINING_SET = 0
DEVELOPMENT_SET = 1
TEST_SET = 2


# In[ ]:


candidates = session.query(DiseaseGene).filter(DiseaseGene.split==TEST_SET)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:


sv

