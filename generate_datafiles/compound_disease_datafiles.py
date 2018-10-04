import pandas as pd
import os

#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)

disease_url = 'https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv'
compound_url = "https://raw.githubusercontent.com/dhimmel/drugbank/7b94454b14a2fa4bb9387cb3b4b9924619cfbd3e/data/drugbank.tsv"
ctpd_url = "https://raw.githubusercontent.com/dhimmel/indications/11d535ba0884ee56c3cd5756fdfb4985f313bd80/catalog/indications.tsv"

base_dir = os.path.join(os.path.dirname(os.getcwd()), 'compound_disease')

disease_ontology_df = pd.read_csv(disease_url, sep="\t")
disease_ontology_df = (
    disease_ontology_df
    .drop_duplicates(["doid_code", "doid_name"])
    .rename(columns={'doid_code': 'doid_id'})
)

drugbank_df = pd.read_table(compound_url).rename(index=str, columns={'name':'drug_name'})

disease_ontology_df["dummy_key"] = 0
drugbank_df["dummy_key"] = 0

pair_df = (
	disease_ontology_df
	.merge(drugbank_df[["drugbank_id", "drug_name", "dummy_key"]], on='dummy_key')
	.drop('dummy_key', axis=1)
	)

compound_treats_palliates_disease_df = (
	pd.read_table(ctpd_url)
	.assign(sources='pharmacotherapydb')
	.drop(["n_curators", "n_resources"], axis=1)
	.rename(index=str, columns={"drug": "drug_name"})
	)

compound_treats_disease_df = (
	compound_treats_palliates_disease_df
	.query("category=='DM'")
	.drop("category", axis=1)
	)

compound_palliates_disease_df = (
	compound_treats_palliates_disease_df
	.query("category=='SYM'")
	.drop("category", axis=1)
	)

ctd_map_df = pair_df.merge(compound_treats_disease_df, how='left')
ctd_map_df['hetionet'] = ctd_map_df.sources.notnull().astype(int)

cpd_map_df = pair_df.merge(compound_palliates_disease_df, how='left')
cpd_map_df['hetionet'] = cpd_map_df.sources.notnull().astype(int)

query = '''
SELECT "Compound_cid" as drugbank_id, "Disease_cid" as doid_id, count(*) AS n_sentences
FROM compound_disease
GROUP BY "Compound_cid", "Disease_cid";
'''
sentence_count_df = pd.read_sql(query, database_str)

ctd_map_df = ctd_map_df.merge(sentence_count_df, how='left')
ctd_map_df.n_sentences = ctd_map_df.n_sentences.fillna(0).astype(int)
ctd_map_df['has_sentence'] = (ctd_map_df.n_sentences > 0).astype(int)

ctd_map_df.to_csv(os.path.join(base_dir, "compound_treats_disease.tsv.xz"), compression='xz', 
	sep="\t", index=False, float_format='%.5g')

cpd_map_df = cpd_map_df.merge(sentence_count_df, how='left')
cpd_map_df.n_sentences = cpd_map_df.n_sentences.fillna(0).astype(int)
cpd_map_df['has_sentence'] = (cpd_map_df.n_sentences > 0).astype(int)

cpd_map_df.to_csv(os.path.join(base_dir, "compound_palliates_disease_df.tsv.xz"), compression='xz', 
	sep="\t", index=False, float_format='%.5g')
