from collections import OrderedDict
import operator

from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, accuracy_score
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix


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

def tag_sentence(x, class_table):
    """
    This function tags the mentions of each candidate sentence.
    x - dataframe with candidate sentences
    class_table - the table for each candidate
    """
    candidates=(
        session
        .query(class_table)
        .filter(class_table.id.in_(x.candidate_id.astype(int).tolist()))
        .all()
    )
    tagged_sen=[
         " ".join(
             mark_sentence(
                candidate_to_tokens(cand), 
                [
                        (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                        (cand[1].get_word_start(), cand[1].get_word_end(), 2)
                ]
            )
         )
        for cand in candidates
    ]

    return tagged_sen


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

def generate_results_df(grid_results, curated_labels, pos_label=1):
    performance_dict = {}
    for lf_sample in grid_results:
        model_param_per = {}
        if isinstance(grid_results[lf_sample], pd.np.ndarray):
            predict_proba = grid_results[lf_sample][:,0]
            precision, recall, _ = precision_recall_curve(
                curated_labels, 
                predict_proba,
                pos_label=pos_label
            )
            fpr, tpr, _ = roc_curve(
                curated_labels, 
                predict_proba,
                pos_label=pos_label
            )
            model_param_per[lf_sample] = [auc(recall, precision), auc(fpr, tpr)]
        else:
            for param, predictions in grid_results[lf_sample].items():
                predict_proba = predictions[:,0]
                precision, recall, _ = precision_recall_curve(
                    curated_labels, 
                    predict_proba,
                    pos_label=pos_label
                )
                fpr, tpr, _ = roc_curve(
                    curated_labels, 
                    predict_proba,
                    pos_label=pos_label
                )
            model_param_per[param] = [auc(recall, precision), auc(fpr, tpr)]
        performance_dict[lf_sample] = max(model_param_per.items(), key=operator.itemgetter(1))[1]
    return pd.DataFrame.from_dict(performance_dict, orient="index")

def embed_word_to_index(cand, word_dict):
    return [word_dict[word] if word in word_dict else 1 for word in cand]

def generate_embedded_df(candidates, word_dict, max_length=83):
    words_to_embed = [
        (
        mark_sentence(
            candidate_to_tokens(cand), 
            [
                    (cand[0].get_word_start(), cand[0].get_word_end(), 1),
                    (cand[1].get_word_start(), cand[1].get_word_end(), 2)
            ]
        ), cand.id)
        for cand in tqdm_notebook(candidates)
    ]
    print(words_to_embed)
    embed_df = pd.SparseDataFrame(
        coo_matrix(
            list(
                map(
                    lambda x: pd.np.pad(
                        embed_word_to_index(x[0], word_dict),
                        (0, (max_length-len(x[0]))),
                        'constant',
                        constant_values=0
                    ),
                    words_to_embed
                )
            )
        ),
        columns=list(range(max_length))
    )
    embed_df['candidate_id'] = list(map(lambda x: x[1], words_to_embed))
    embed_df['sen_length'] = list(map(lambda x: len(x[0]), words_to_embed))
    return embed_df