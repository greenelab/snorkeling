import pandas as pd
from tqdm import tqdm_notebook

#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)

compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
cbg_url = "https://raw.githubusercontent.com/dhimmel/integrate/93feba1765fbcd76fd79e22f25121f5399629148/compile/CbG-binding.tsv"
full_map_output_file = "data/compound_gene/compound_binds_gene/compound_gene_pairs_binds_full_map.csv"
full_map_sen_count_file = "data/compound_gene/compound_binds_gene/compound_gene-pairs_binds_mapping.csv"
final_output_file = "data/compound_gene/compound_binds_gene/compound_gene_pairs_binds.csv"


compound_df = pd.read_table(compound_url)

gene_entrez_df = (
    pd.read_table(gene_url, dtype={'GeneID': str}) [["GeneID", "Symbol"]]
    .rename(columns={'GeneID': 'entrez_gene_id', 'Symbol': 'gene_symbol'})
)

gene_entrez_df['dummy_key'] = 0
compound_df['dummy_key'] = 0
pair_df = (
	gene_entrez_df
	.merge(compound_df[["drugbank_id", "name", "dummy_key"]], on='dummy_key')
	.drop('dummy_key', axis=1)
)
pair_df.to_csv(full_map_output_file, index=False, float_format='%.5g')

#release memory
del pair_df

compound_binds_gene_df = pd.read_table(cbg_url, dtype={'entrez_gene_id': int})

query = '''
SELECT "Compound_cid" AS drugbank_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM compound_gene
GROUP BY "Compound_cid", "Gene_cid";
'''
sentence_count_df = (
    pd.read_sql(query, database_str)
    .astype(dtype={'entrez_gene_id': int})
)

for r in tqdm_notebook(pd.read_csv("data/compound_gene/compound_binds_gene/compound_gene_pairs_binds_full_map.csv", chunksize=1e6, dtype={'entrez_gene_id': int})):
    merged_df = pd.merge(r, compound_binds_gene_df[["drugbank_id", "entrez_gene_id", "sources"]], how="left")
    merged_df['hetionet'] = merged_df.sources.notnull().astype(int)
    merged_df = merged_df.merge(sentence_count_df, how='left', copy=False)
    merged_df.n_sentences = merged_df.n_sentences.fillna(0).astype(int)
    merged_df['has_sentence'] = (merged_df.n_sentences > 0).astype(int)
    merged_df.to_csv(full_map_sen_count_file, mode='a', index=False)


# Memory issues occur when I try to build the full dataframe
# Have to rely on command line to remedy this issue
os.system(
    "head -n 1 {}  > {};".format(full_map_output_file, full_map_sen_count_file) +
    "cat {}} |  awk -F ',' '{if($8==1) print $0}' >> {}".format(full_map_sen_count_file, final_output_file)
)

cbg_map_df = pd.read_csv(final_output_file)
