import pandas as pd
import numpy as np
from scipy.stats import norm
import scipy.sparse as sparse

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