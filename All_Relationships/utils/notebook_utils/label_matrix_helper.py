def get_columns(session, L_data, lf_hash, lf_name):
    """
    This function is designed to extract the column positions of
    each individual label function given their category (i.e. CbG_DB or DaG_TEXT ...)
    
    returns a list of column positions that corresponds to each label function 
    """
    return list(
    map(lambda x: L_data.key_index[x[0]],
    session.query(L_data.annotation_key_cls.id)
         .filter(L_data.annotation_key_cls.name.in_(list(lf_hash[lf_name].keys())))
         .all()
        )
    )
