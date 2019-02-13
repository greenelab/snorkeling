import threading
import queue
import sys
import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.sparse as sparse
from tqdm import tqdm_notebook

from snorkel.models import Candidate

candidate_queue = queue.Queue()
data_queue = queue.Queue()

def get_columns(session, L_data, lf_hash, lf_name):
    """
    This function is designed to extract the column positions of
    each individual label function given their category (i.e. CbG_DB or DaG_TEXT ...)

    L_data - the sparse label matrix
    lf_hash - a dictionary containing label functions groupd into specific categories
        the keys are label function groups and the valeus are  dictionaries containg 
        label function names (keys) to their acutal fucntions (values)
    lf_name - the  query for the label function category of interest
    
    returns a list of column positions that corresponds to each label function 
    """
    return list(
    map(lambda x: L_data.key_index[x[0]],
    session.query(L_data.annotation_key_cls.id)
         .filter(L_data.annotation_key_cls.name.in_(list(lf_hash[lf_name].keys())))
         .all()
        )
    )


def get_auc_significant_stats(data_df, model_aucs):
    """
    This function is designed test the hypothesis that 
    given aurocs are greater than 0.5 (random)

    relevant websites:
     https://www.quora.com/How-is-statistical-significance-determined-for-ROC-curves-and-AUC-values
     https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test


    data_df - a dataframe that contains sentences that were hand labeded 
        for validation purposes.

    model_aucs - dictionary that contains the names of models as the key
        and the auroc score as the values.

    returns a dataframe with the provided statsitics
    """

    class_df = data_df.curated_dsh.value_counts()
    n1 = class_df[0]
    n2 = class_df[1]
    mu = (n1*n2)/2
    sigma_u = np.sqrt((n1 * n2 * (n1+n2+1))/12)
    print("mu: {:f}, sigma: {:f}".format(mu, sigma_u))

    model_auc_df = (
        pd.DataFrame
        .from_dict(model_aucs, orient='index')
        .rename(index=str, columns={0:'auroc'})
    )

    model_auc_df['u'] = model_auc_df.auroc.apply(lambda x: x*n1*n2)
    model_auc_df['z_u'] = model_auc_df.u.apply(lambda z_u: (z_u- mu)/sigma_u)
    model_auc_df['p_value'] = model_auc_df.z_u.apply(lambda z_u: norm.sf(z_u, loc=0, scale=1))

    return model_auc_df


def get_overlap_matrix(L, normalize=False):
    """
    This code is "borrowed" from the snorkel metal repo.
    It is designed to output a matrix of overlaps between label fucntions

    L - a sparse label matrix  created by snorkel.annotations.LabelAnnotator, 
        contains output from each label function and extract information 
        such as names of the label functions

    returns a matrix that contains the overlaps between label functions
    """
    L = L.todense() if sparse.issparse(L) else L
    n, m = L.shape
    X = np.where(L != 0, 1, 0).T
    G = X @ X.T

    if normalize:
        G = G / n
    return G

def get_conflict_matrix(L, normalize=False):
    """
    This code is "borrowed" from the snorkel metal repo.
    It is designed to output a matrix of overlaps between label fucntions

    L - a sparse label matrix  created by snorkel.annotations.LabelAnnotator, 
        contains output from each label function and extract information 
        such as names of the label functions

    returns a matrix that contains the conflicts between label functions
    """
    L = L.todense() if sparse.issparse(L) else L
    n, m = L.shape
    C = np.zeros((m, m))

    # Iterate over the pairs of LFs
    for i in range(m):
        for j in range(m):
            # Get the overlapping non-zero indices
            overlaps = list(
                set(np.where(L[:, i] != 0)[0]).intersection(
                    np.where(L[:, j] != 0)[0]
                )
            )
            C[i, j] = np.where(L[overlaps, i] != L[overlaps, j], 1, 0).sum()

    if normalize:
        C = C / n
    return C

def label_candidates_db(labeler, cids_query, label_functions, apply_existing=False):
    """
    This function is designed to label candidates and place the annotations inside a database.
    Will be rarely used since snorkel metal doesn't use a database for annotations.
    Important to keep if I were to go back towards snorkel's original database version

    labeler - the labeler object
    cids_query - the query make for extracting candidate objects
    label_functions - a list of label functions to generate annotationslf_stats
    """
    if apply_existing:
        return labeler.apply_existing(cids_query=cids_query, parllelistm=5, clear=False)
    else:
        return labeler.apply(cids_query=cids_query, parallelism=5)


def label_candidates(session, candidate_ids, lfs, lf_names, multitask=False, num_threads=4, batch_size=10):
    """
    This function returns a sparse matrix in memory. Helps bypass using a static database to store annotations
    Only catch is that this structure doesn't contain the names of label functions
    
    session - the session object
    candidate_ids - the ids for candidates to be extracted
    lfs - a list of label functions to label candidates
    multitask - a boolean to signify that labels will be in multitask format
    num_threads - the number of threads to execute
    batch_size - the batch size to prevent a memory overload
    """
    if multitask == True:
        raise Exception("Must Fix index error for multitask version")
    iteration = 0
    started = False
    
    #set up the threads
    thread_list = [
    threading.Thread(target=_label_candidates_multithread, args=(lfs, multitask))
    for t in range(num_threads) 
    ]
    
    candidate_map = {}
    with tqdm_notebook(total=len(candidate_ids)) as pbar:
        while iteration < len(candidate_ids):
            candidate_batch = candidate_ids[iteration:iteration + batch_size]
            row_indicies = range(iteration, iteration + batch_size, 1)

            #Fire up the threads to label each candidate
            candidates = session.query(Candidate).filter(Candidate.id.in_(candidate_batch)).all()
            for row_index, cand_obj in zip(row_indicies, candidates):
                candidate_queue.put((row_index, cand_obj))
                candidate_map[row_index] = cand_obj.id

            #Only start the threads once
            if not(started):
                for thread in thread_list:
                    thread.start()
                started = True

            pbar.update(batch_size)
            iteration += batch_size

    for thread in thread_list:
        thread.join()


    # Once finished use appropiate 
    # steps to form the matricies
    if multitask:
        multirow  = [[] for i in lfs]
        multicol = [[] for i in lfs]
        multidata = [[] for i in lfs]
        candidate_entries = []

        while not(data_queue.empty()):
            entry = data_queue.get()
            multirow[entry[0]].append(entry[1])
            multicol[entry[0]].append(entry[2])
            multidata[entry[0]].append(entry[3])
            if entry[4] not in candidate_entries:
                candidate_entries.append(entry[4])

        L_data = [
            pd.SparseDataFrame(
        sparse.csr_matrix(
            (
                multidata[task_index], (multirow[task_index], multicol[task_index])),
                shape=(len(candidate_ids), max([len(lf) for lf in lfs]))
            ))
        for task_index in range(len(lfs))
        ]
        L_data[0]['candidate_id'] = candidate_entries
        L_data[1]['candidate_id'] = candidate_entries
        L_data[2]['candidate_id'] = candidate_entries
        return L_data
    else:
        row = []
        data =[]
        col = []

        candidate_entries = (
            pd.DataFrame(list(candidate_map.items()), columns=["row_index", "candidate_id"])
            .sort_values("row_index")
        )

        while not(data_queue.empty()):
            entry = data_queue.get()

            row.append(entry[0])
            col.append(entry[1])
            data.append(entry[2])

        label_matrix = sparse.csr_matrix((data, (row, col)), shape=(len(candidate_ids),len(lfs)))
        label_matrix_df = pd.SparseDataFrame(label_matrix, columns=lf_names)
        label_matrix_df['candidate_id'] = candidate_entries['candidate_id'].values.tolist()
        
        return label_matrix_df

def _label_candidates_multithread(lfs, multitask):
    """
    This function is called when each thread is created.

    lfs - the label functions to annotate candidates
    multitask - a boolean that tells the function to label candidates in a multitask format
    """
    while not(candidate_queue.empty()):
       
        candidate = candidate_queue.get()
        sys.stdout.write("\r{:7d}".format(candidate_queue.qsize()))
        sys.stdout.flush()
        
        if multitask:
            for task_index, lf_task in enumerate(lfs):
                for col_index, lf in enumerate(lf_task):
                    val = lf(candidate[1])
                    
                    if val != 0:
                        data_queue.put((task_index, candidate[0], col_index, val))
        else:
            for col_index, lf in enumerate(lfs):
                val = lf(candidate[1])
            
                if val != 0:
                    # put row_index, col_index and data onto a synchronized queue
                    data = (candidate[0], col_index, val)
                    data_queue.put(data)
    
