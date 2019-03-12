from collections import OrderedDict
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np


def create_gen_marginal_df(L_data, models, lfs_columns, model_names, candidate_ids):
    """
    This function is designed to create a dataframe that will hold
    the marginals outputted from the generative model

    L_data - the sparse matrix generated fromt eh label functions
    models - the list of generative models
    lfs_columns - a listing of column indexes that correspond to desired label fucntions
    model_names - a label for each model
    candidate_ids - a list of candidate ids so the marginals can be mapped back to the candidate
    """
    marginals = [
        model.marginals(L_data[:, columns])
        for model, columns in zip(models, lfs_columns)
    ]
    marginals_df = pd.DataFrame(
           np.array(marginals).T, columns=model_names
    )
    marginals_df['candidate_id'] = candidate_ids
    return marginals_df


def create_disc_marginal_df(models, test_data):
    """
    This function is desgined get the predicted marginals from the sklearn models

    models - list of sklearn models that marginals will be generated from

    test_data - the dev set data used to generate testing marginals

    return a dataframe containing marginal probabilities for each sklearn model
    """

    return (
        pd.DataFrame([model.best_estimator_.predict_proba(test_data)[:,1] for model in models])
        .transpose()
        .rename(index=str, columns=columns)
        )

# Taken from hazyresearch/snorkel repository
# https://github.com/HazyResearch/snorkel/blob/2866e45f03b363032cd11117f59f99803233c739/snorkel/learning/pytorch/rnn/utils.py
def scrub(s):
    return ''.join(c for c in s if ord(c) < 128)

# Taken from hazyresearch/snorkel repository
# https://github.com/HazyResearch/snorkel/blob/2866e45f03b363032cd11117f59f99803233c739/snorkel/learning/pytorch/rnn/utils.py
def candidate_to_tokens(candidate, token_type='words'):
    tokens = candidate.get_parent().__dict__[token_type]
    return [scrub(w).lower() for w in tokens]

# Taken from hazyresearch/snorkel repository
# https://github.com/HazyResearch/snorkel/blob/2866e45f03b363032cd11117f59f99803233c739/snorkel/learning/pytorch/rnn/rnn_base.py
def mark(l, h, idx):
    """Produce markers based on argument positions
    
    :param l: sentence position of first word in argument
    :param h: sentence position of last word in argument
    :param idx: argument index (1 or 2)
    """
    return [(l, "{}{}".format('~~[[', idx)), (h+1, "{}{}".format(idx, ']]~~'))]

# Taken from hazyresearch/snorkel repository
# https://github.com/HazyResearch/snorkel/blob/2866e45f03b363032cd11117f59f99803233c739/snorkel/learning/pytorch/rnn/rnn_base.py
def mark_sentence(s, args):
    """Insert markers around relation arguments in word sequence
    
    :param s: list of tokens in sentence
    :param args: list of triples (l, h, idx) as per @_mark(...) corresponding
               to relation arguments
    
    Example: Then Barack married Michelle.  
         ->  Then ~~[[1 Barack 1]]~~ married ~~[[2 Michelle 2]]~~.
    """
    marks = sorted([y for m in args for y in mark(*m)], reverse=True)
    x = list(s)
    for k, v in marks:
        x.insert(k, v)
    return x

def make_sentence_df(candidates):
    """ 
    This function creats a dataframe for all candidates (sentences that contain at least two mentions)
    located in our database.
    
    candidates - a list of candidate objects passed in from sqlalchemy

    return a Dataframe that contains each candidate sentence  and the corresponding candidate entities
    """
    rows = list()
    for c in tqdm_notebook(candidates):
        args = [
                (c[0].get_word_start(), c[0].get_word_end(), 1),
                (c[1].get_word_start(), c[1].get_word_end(), 2)
            ]
        sen = " ".join(mark_sentence(candidate_to_tokens(c), args))
        if hasattr(c, 'Disease_cid') and hasattr(c, 'Gene_cid'):
            row = OrderedDict()
            row['candidate_id'] = c.id
            row['disease'] = c[0].get_span()
            row['gene'] = c[1].get_span()
            row['doid_id'] = c.Disease_cid
            row['entrez_gene_id'] = c.Gene_cid
            row['sentence'] = sen
        elif hasattr(c, 'Gene1_cid') and hasattr(c, 'Gene2_cid'):
            row = OrderedDict()
            row['candidate_id'] = c.id
            row['gene1'] = c[0].get_span()
            row['gene2'] = c[1].get_span()
            row['gene1_id'] = c.Gene1_cid
            row['gene2_id'] = c.Gene2_cid
            row['sentence'] = sen
        elif hasattr(c, 'Compound_cid') and hasattr(c, 'Gene_cid'):
            row = OrderedDict()
            row['candidate_id'] = c.id
            row['compound'] = c[0].get_span()
            row['gene'] = c[1].get_span()
            row['drugbank_id'] = c.Compound_cid
            row['entrez_gene_id'] = c.Gene_cid
            row['sentence'] = sen
        elif hasattr(c, 'Compound_cid') and hasattr(c, 'Disease_cid'):
            row = OrderedDict()
            row['candidate_id'] = c.id
            row['compound'] = c[0].get_span()
            row['disease'] = c[1].get_span()
            row['drugbank_id'] = c.Compound_cid
            row['doid_id'] = c.Disease_cid
            row['sentence'] = sen
        rows.append(row)
    return pd.DataFrame(rows)
        

def write_candidates_to_excel(candidate_df, spreadsheet_name):
    """
    This function is designed to save the candidates to an excel
    spreadsheet. This is needed for manual curation of candidate 
    sentences
    
    candidate_df - the dataframe that holds all the candidates
    spreadsheet_name - the name of the excel spreadsheet
    """
    writer = pd.ExcelWriter(spreadsheet_name)
    (
        candidate_df
        .to_excel(writer, sheet_name='sentences', index=False)
    )
    if writer.engine == 'xlsxwriter':
        for sheet in writer.sheets.values():
            sheet.freeze_panes(1, 0)
    writer.close()
    return

def load_candidate_dataframes(filename, curated_field):
    """
    This function reads in the candidates excel files to preform analyses.

    dataframe - the path of the dataframe to load
    """
    data_df = pd.read_excel(filename)
    data_df = data_df.query("{}.notnull()".format(curated_field))
    return data_df.sort_values('candidate_id')
