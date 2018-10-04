import pandas as pd
import tqdm

#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)

compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
cbg_url = "https://raw.githubusercontent.com/dhimmel/integrate/93feba1765fbcd76fd79e22f25121f5399629148/compile/CbG-binding.tsv"
crg_url = "https://raw.githubusercontent.com/dhimmel/lincs/bbc6812b7d19e98637b44373cdfc52f61bce6327/data/consensi/signif/dysreg-drugbank.tsv"

base_dir = os.path.join(os.path.dirname(os.getcwd()), 'compound_gene')

full_map_output_file = os.path.join(base_dir, "compound_gene_pairs.csv")

cbg_sen_count_file = os.path.join(base_dir, "compound_gene-pairs_binds_sen_count.csv")
cug_sen_count_file = os.path.join(base_dir, "compound_gene-pairs_upreg_sen_count.csv")
cdg_sen_count_file = os.path.join(base_dir, "compound_gene-pairs_downreg_sen_count.csv")

final_cbg_output_file = os.path.join(base_dir, "compound_binds_gene/compound_gene_pairs_binds.csv")
final_cug_output_file = os.path.join(base_dir, "compound_gene_pairs_upregulates.csv")
final_cdg_output_file = os.path.join(base_dir, "compound_gene_pairs_downregulates.csv")


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

compound_regulates_gene_df = (
    pd.read_table(crg_url, dtype={'entrez_gene_id': int})
    .assign(sources='lincs')
    .drop(['z_score', 'status', 'nlog10_bonferroni_pval'], axis=1)
    .rename(index=str, columns={"perturbagen":'drugbank_id'})
    )

compound_upregulates_gene_df  = (
    compound_regulates_gene_df.query("direction == 'up'").drop('direction', axis=1)
    )

compound_downregulates_gene_df = (
    compound_regulates_gene_df.query("direction == 'down'").drop('direction', axis=1)
    )

query = '''
SELECT "Compound_cid" AS drugbank_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM compound_gene
GROUP BY "Compound_cid", "Gene_cid";
'''

sentence_count_df = (
    pd.read_sql(query, database_str)
    .astype(dtype={'entrez_gene_id': int})
)

for r in tqdm.tqdm(pd.read_csv(full_map_output_file, chunksize=1e6, dtype={'entrez_gene_id': int})):
    merged_df = pd.merge(r, compound_binds_gene_df[["drugbank_id", "entrez_gene_id", "sources"]], how="left")
    merged_df['hetionet'] = merged_df.sources.notnull().astype(int)
    merged_df = merged_df.merge(sentence_count_df, how='left', copy=False)
    merged_df.n_sentences = merged_df.n_sentences.fillna(0).astype(int)
    merged_df['has_sentence'] = (merged_df.n_sentences > 0).astype(int)
    merged_df.to_csv(cbg_sen_count_file, mode='a', index=False)

    merged_df = pd.merge(r, compound_upregulates_gene_df[["drugbank_id", "entrez_gene_id", "sources"]], how="left")
    merged_df['hetionet'] = merged_df.sources.notnull().astype(int)
    merged_df = merged_df.merge(sentence_count_df, how='left', copy=False)
    merged_df.n_sentences = merged_df.n_sentences.fillna(0).astype(int)
    merged_df['has_sentence'] = (merged_df.n_sentences > 0).astype(int)
    merged_df.to_csv(cug_sen_count_file, mode='a', index=False)

    merged_df = pd.merge(r, compound_downregulates_gene_df[["drugbank_id", "entrez_gene_id", "sources"]], how="left")
    merged_df['hetionet'] = merged_df.sources.notnull().astype(int)
    merged_df = merged_df.merge(sentence_count_df, how='left', copy=False)
    merged_df.n_sentences = merged_df.n_sentences.fillna(0).astype(int)
    merged_df['has_sentence'] = (merged_df.n_sentences > 0).astype(int)
    merged_df.to_csv(cdg_sen_count_file, mode='a', index=False)

# Memory issues occur when I try to build the full dataframe
# Have to rely on command line to remedy this issue
os.system(
    "head -n 1 {}  > {};".format(full_map_output_file, cbg_sen_count_file) +
    "cat {}} |  awk -F ',' '{if($8==1) print $0}' >> {}".format(cbg_sen_count_file, final_cbg_output_file)
)

os.system(
    "head -n 1 {}  > {};".format(full_map_output_file, cug_sen_count_file) +
    "cat {}} |  awk -F ',' '{if($8==1) print $0}' >> {}".format(cug_sen_count_file, final_cug_output_file)
)

os.system(
    "head -n 1 {}  > {};".format(full_map_output_file, cdg_sen_count_file) +
    "cat {}} |  awk -F ',' '{if($8==1) print $0}' >> {}".format(cdg_sen_count_file, final_cdg_output_file)
)
