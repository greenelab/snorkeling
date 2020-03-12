#!/usr/bin/env python
# coding: utf-8

# # Inserting Pubtator into a Database

# This notebook is designed to load each [Pubmed](https://www.ncbi.nlm.nih.gov/pubmed/) abstract. It uses our own [pubtator repository](https://github.com/greenelab/pubtator) to convert each abstract into the appropriate format (xml) to load into a postgres database. Run the pubtator scripts before running this notebook, so the pubtator xml file can be constructed. Once constructed this notebook is designed to parse the data into a database. After loading each abstract, candidate extraction is performed for each relationship type.

# ## MUST RUN AT THE START OF EVERYTHING

# Load all necessary imports for the rest of this notebook. Plus, set up the postgres database for database operations. 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

#Imports
import csv
import os
import random

import numpy as np
import pandas as pd
import tqdm


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


from snorkel.candidates import PretaggedCandidateExtractor
from snorkel.models import Document, Sentence, candidate_subclass
from snorkel.parser import CorpusParser
from snorkel.viewer import SentenceNgramViewer
from snorkel.models import Document

from sqlalchemy import func
from string import punctuation
import lxml.etree as et
from database_insertion import *


# # Parse the Pubmed Abstracts

# The code below is designed to read and parse data gathered from our [pubtator repo](https://github.com/greenelab/pubtator) (will refer as PubtatorR for clarity). PubtatorR is designed to gather and parse [NER](https://en.wikipedia.org/wiki/Named-entity_recognition) tagged Medline abstracts from NCBI's [PubTator](https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/PubTator/). It outputs PubTator's annotated text in xml format, which is the standard format we are going to use. For this project, the id's are not from PubTator but from [Hetionet](https://think-lab.github.io/p/rephetio/). Since Pubtator contains over 10 million abstracts, the code below and in subsequent notebooks have been optimized to be memory efficient.

# In[ ]:


filter_df = pd.read_table('https://github.com/greenelab/pubtator/raw/631e86002e11c41cfcfb0043e60b84ab321bdae3/data/pubtator-hetnet-tags.tsv.xz')


# In[ ]:


grouped = filter_df.groupby('pubmed_id')


# In[ ]:


# Please change to your local document here
# Refer to https://github.com/greenelab/pubtator for instructions
# to download and parse Pubtator
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
    # scale properly
    if len(document_chunk) >= 5e4:
        corpus_parser.apply(document_chunk, parallelism=5, clear=False)
        document_chunk = []
    
# If generator exhausts and there are still
# document to parse
if len(document_chunk) > 0:
    corpus_parser.apply(data, parallelism=5, clear=False)
    document_chunk = []


# # Get each candidate relation

# After parsing the above abstracts, the next step in this pipeline is to extract candidates from all the tagged sentences. A candidate is considered a candidate if two mentions occur in the same sentence. For this pilot study, we are only considering the follow candidate relationships: Disease-Gene, Gene-Gene, Compound-Gene, Compound-Disease. In conjunction with extracting candidates, this part of the pipeline also stratifies each sentence into three different categories: Train (70%), Dev (20%), and Test (10%). These set categories will be used in subsequent notebooks ([3](3.data-gen-model.ipynb), [4](4.data-disc-model.ipynb), [5](5.data-analysis.ipynb)) for training and testing the machine learning algorithms.

# In[ ]:


chunk_size = 2e5


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
total_sentences = session.execute("select count(*) from sentence").fetchone()[0]


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
            category_index = category_index + 1
            
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


print_candidates(session, DiseaseGene, 'DiseaseGene')
print_candidates(session, GeneGene, 'GeneGene')
print_candidates(session, CompoundGene, 'CompoundGene')
print_candidates(session, CompoundDisease, 'CompoundDisease')


# # Look at the Potential Candidates

# The one cool thing about jupyter is that you can use this tool to look at candidates. Check it out after everything above has finished running. The highlighted words are what Hetionet tagged as name entities.

# In[ ]:


TRAINING_SET = 0
DEVELOPMENT_SET = 1
TEST_SET = 2


# In[ ]:


candidates = session.query(DiseaseGene).filter(DiseaseGene.split==TRAINING_SET).limit(100)
sv = SentenceNgramViewer(candidates, session)


# In[ ]:


sv

