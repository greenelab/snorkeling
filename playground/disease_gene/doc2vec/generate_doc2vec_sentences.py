import glob
import tqdm
import pandas as pd
import os
import sys

import argparse

sys.path.append(os.path.abspath('../../../modules'))

from utils.notebook_utils.dataframe_helper import load_candidate_dataframes
from utils.notebook_utils.doc2vec_helper import get_candidate_objects, execute_doc2vec, write_sentences_to_file

#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()

from snorkel.models import Candidate, candidate_subclass
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])

#Set up the argparser

parser=argparse.ArgumentParser(description='Run Doc2Vec on sentences')
parser.add_argument('all_dg', desc='Use this flag to embed sentences using all DG sentences', action='store_true')
parser.add_argument('500k', desc='Use this flag to embed sentences using 500k randomly sampled DG sentences', action='store_true')

args=parser.parse_args()


#Run the program
print("Writing sentences to be embedded")
spreadsheet_names = {
    'train': '../../sentence_labels_train.xlsx',
    'dev': '../../sentence_labels_train_dev.xlsx',
    'test': '../../sentence_labels_dev.xlsx'
}

candidate_dfs = {
    key:load_candidate_dataframes(spreadsheet_names[key])
    for key in spreadsheet_names
}

sentence_files = {
    'train':'../../doc2vec/embedding_sentences/train_sentences.txt',
    'dev': '../../doc2vec/embedding_sentences/dev_sentences.txt',
    'test':'../../doc2vec/embedding_sentences/test_sentences.txt'
}

write_sentences_to_file(
    get_candidate_objects(session, candidate_dfs),
    sentence_files
)

if args.all_dg:

    offset = 0
    batch_size = 50000
    sql_df = pd.read_sql('''SELECT id AS candidate_id FROM disease_gene''', database_str)
    with open('doc2vec/training_sentences/all_dg_sentences.txt', 'w') as f:
        while True:
            cand_ids = sql_df.iloc[offset:offset+batch_size]
            
            if cand_ids.empty:
                break
                
            cands = (
                session
                .query(Candidate)
                .filter(Candidate.id.in_(cand_ids.candidate_id.astype(int).tolist()))
                .all()
            )
        
            for c in tqdm.tqdm(cands):
                    f.write(c.get_parent().text + "\n")

            offset += 50000

    execute_doc2vec(
        '../../doc2vec/training_sentences/all_dg_sentences.txt', 
        '../../doc2vec/word_vectors/train_all_dg_word_vectors.txt',
        '../../doc2vec/doc_vectors/train_doc_vectors_all_dg.txt',
        '../../doc2vec/embedding_sentences/train_sentences.txt',
        '../../doc2vec/vocab/train_vocab_all_dg.txt'
    )
    execute_doc2vec(
        '../../doc2vec/training_sentences/all_dg_sentences.txt', 
        '../../doc2vec/word_vectors/dev_all_dg_word_vectors.txt',
        '../../doc2vec/doc_vectors/dev_doc_vectors_all_dg.txt',
        '../../doc2vec/embedding_sentences/dev_sentences.txt',
        '../../doc2vec/vocab/train_vocab_all_dg.txt',
        read_vocab=True
    )
    execute_doc2vec(
        '../../doc2vec/training_sentences/all_dg_sentences.txt', 
        '../../doc2vec/word_vectors/test_all_dg_word_vectors.txt',
        '../../doc2vec/doc_vectors/test_doc_vectors_all_dg.txt',
        '../../doc2vec/embedding_sentences/test_sentences.txt',
        '../../doc2vec/vocab/train_vocab_all_dg.txt',
        read_vocab=True
    )

if args.500k:
    iterations = 10

    for index in range(iterations):
        query='''
        SELECT id AS candidate_id FROM disease_gene
        ORDER BY RANDOM()
        LIMIT 500000
        ''' 
            
        data_df = pd.read_sql(query, database_str).sort_values('candidate_id')
        output_file = 'doc2vec/training_sentences/dg_500k_subset_{}.txt'.format(index)

        batch_size=50000
        offset = 0
        with open(output_file, 'w') as f:
            while True:
                cand_ids = data_df.iloc[offset:offset+batch_size]
            
                if cand_ids.empty:
                    break

                cands = (
                    session
                    .query(Candidate)
                    .filter(Candidate.id.in_(cand_ids.candidate_id.astype(int).tolist()))
                    .all()
                )

                for c in tqdm.tqdm(cands):
                    f.write(c.get_parent().text + "\n")

                offset += 50000


    for subset, file_name in tqdm_notebook(enumerate(glob.glob('../../doc2vec/training_sentences/dg_500k_subset_*.txt'))):
        execute_doc2vec(
            file_name, 
            '../../doc2vec/word_vectors/train_word_vectors_500k_subset_{}.txt'.format(subset),
            '../../doc2vec/doc_vectors/train_doc_vectors_500k_subset_{}.txt'.format(subset),
            '../../doc2vec/embedding_sentences/train_sentences.txt',
            '../../doc2vec/vocab/train_word_vectors_500k_subset_{}.txt'.format(subset)
        )
        execute_doc2vec(
            file_name, 
            '../../doc2vec/word_vectors/dev_word_vectors_500k_subset_{}.txt'.format(subset),
            '../../doc2vec/doc_vectors/dev_doc_vectors_500k_subset_{}.txt'.format(subset),
            '../../doc2vec/embedding_sentences/dev_sentences.txt',
            '../../doc2vec/vocab/train_word_vectors_500k_subset_{}.txt'.format(subset),
            read_vocab=True
        )
        execute_doc2vec(
            file_name, 
            '../../doc2vec/word_vectors/test_word_vectors_500k_subset_{}.txt'.format(subset),
            '../../doc2vec/doc_vectors/test_doc_vectors_500k_subset_{}.txt'.format(subset),
            '../../doc2vec/embedding_sentences/test_sentences.txt',
            '../../doc2vec/vocab/train_word_vectors_500k_subset_{}.txt'.format(subset),
            read_vocab=True
        )