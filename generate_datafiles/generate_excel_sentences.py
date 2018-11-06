import os
import sys

import pandas as pd

#Set up the path so my module scripts can be imported
sys.path.append(os.path.abspath('../modules'))

# This module is designed to help create dataframes for each candidate sentence
from utils.notebook_utils.dataframe_helper import make_sentence_df, write_candidates_to_excel

#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()

#Import the candidate class for postgres extraction
from snorkel.models import candidate_subclass
from snorkel.models import Candidate

#Define the polymorphic class that snorkel uses
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
GeneGene = candidate_subclass('GeneGene', ['Gene1', 'Gene2'])
CompoundGene = candidate_subclass('CompoundGene', ['Compound', 'Gene'])
CompoundDisease = candidate_subclass('CompoundDisease', ['Compound', 'Disease'])

#Name of the spreadsheets that will be outputted
spreadsheet_names = {
    'train': 'sentence_labels_train.xlsx',
    'dev': 'sentence_labels_dev.xlsx',
    'test': 'sentence_labels_test.xlsx'
}

#The sql statements used to extract sentences from the database
sql_statements = [
    '''
    SELECT id from candidate
    WHERE split = 0 and type='disease_gene'
    ORDER BY RANDOM()
    LIMIT 50000;
    ''',
    
    '''
    SELECT id from candidate
    WHERE split = 1 and type='disease_gene'
    ORDER BY RANDOM()
    LIMIT 10000;
    ''',

    '''
    SELECT id from candidate
    WHERE split = 2 and type='disease_gene'
    ORDER BY RANDOM()
    LIMIT 10000;
    '''
]

#Exectue the queries and output them to excel files.
session.execute("SELECT setseed(0.5);")
for sql, spreadsheet_name in zip(sql_statements, spreadsheet_names.values()):
    target_cids = [x[0] for x in session.execute(sql)]
    candidates = (
        session
        .query(Candidate)
        .filter(Candidate.id.in_(target_cids))
        .all()
    )
    candidate_df = make_sentence_df(candidates)
    write_candidates_to_excel(candidate_df, spreadsheet_name)