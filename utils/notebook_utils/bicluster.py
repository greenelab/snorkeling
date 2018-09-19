import pandas as pd
import re

def fix(x):
    """
    This function is designed as a quick fix
    for the dependency path problem. This problem is that the START_ENTITY and END_ENTIY 
    are capitalized in part II files, but not in part 1 files. If
    not fixed, it would cause a mapping error.

    x - the string to chagne from upper case to lower case.
    """
    x = re.sub("START_ENTITY", "start_entity", x)
    x = re.sub("END_ENTITY", "end_entity", x)
    return x

def create_bicluster_df(dep_file, dep_dist_file, output_df_file):
    """
    Designed to create a dataframe of all the dependency paths mentioned in
    https://zenodo.org/record/1035500#.W4l6IhgpBrk

    dep_file - name of the dependency path file
    dep_dist_file - name of the dep distribution file
    output_df_file - the output dataframe file
    """
    named_columns = [
    "pubmed_id", "sentence_num", "first_entity",
    "first_entity_location", "second_entity", "second_entity_location",
    "first_entity_string", "second_entity_string", "first_entity_db_id",
    "second_entity_db_id", "first_entity_type", "second_entity_type", 
    "dependency_path", "sentence_string"
    ]

    dist_df = pd.read_table(dep_dist_file)

    dep_df = pd.read_table(dep_file, names=named_columns)
    dep_df.dependency_path = dep_df.dependency_path.apply(fix)

    (
        dep_df[["pubmed_id", "sentence_num", "dependency_path", "sentence_string"]]
        .merge(dist_df, left_on="dependency_path", right_on="path")
        .to_csv(output_df_file, sep="\t", index=False, compression='xz')
    )
    return