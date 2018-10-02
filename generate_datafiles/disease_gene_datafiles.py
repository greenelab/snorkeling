import pandas as pd

#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)

disease_url = "https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv"
gene_url = "https://raw.githubusercontent.com/dhimmel/entrez-gene/a7362748a34211e5df6f2d185bb3246279760546/data/genes-human.tsv"
dag_url = "https://github.com/dhimmel/integrate/raw/93feba1765fbcd76fd79e22f25121f5399629148/compile/DaG-association.tsv"
drg_url = "https://raw.githubusercontent.com/dhimmel/stargeo/08b126cc1f93660d17893c4a3358d3776e35fd84/data/diffex.tsv"

disease_ontology_df = (
	pd.read_csv(disease_url, sep="\t")
	.drop_duplicates(["doid_code", "doid_name"])
	.rename(columns={'doid_code': 'doid_id'})
)

gene_entrez_df = (
	pd.read_table(gene_url, dtype={'GeneID': str})[["GeneID", "Symbol"]]
    .rename(columns={'GeneID': 'entrez_gene_id', 'Symbol': 'gene_symbol'})
)

disease_ontology_df['dummy_key'] = 0
gene_entrez_df['dummy_key'] = 0

pair_df = (
	gene_entrez_df
	.merge(disease_ontology_df[["doid_id", "doid_name", "dummy_key"]], on='dummy_key')
	.drop('dummy_key', axis=1)
)

disease_associates_gene_df = (
	pd.read_table(dag_url, dtype={'entrez_gene_id': str})
	)

disease_regulates_gene_df = (
	pd.read_table(drg_url, dtype={'entrez_gene_id': str})
	.assign(sources='strego')
	.rename(index=str, columns={'slim_id':'doid_id', 'slim_name':'doid_name'})
	.drop(["log2_fold_change", "p_adjusted"], axis=1)
	)

disease_downregulates_gene_df = disease_regulates_gene_df.query("direction=='down'").drop('direction', axis=1)
disease_upregulates_gene_df = disease_regulates_gene_df.query("direction=='up'").drop('direction', axis=1)

dag_map_df = pair_df.merge(disease_associates_gene_df, how='left')
dag_map_df['hetionet'] = dag_map_df.sources.notnull().astype(int)

dug_map_df =  pair_df.merge(disease_upregulates_gene_df, how='left')
dug_map_df['hetionet'] = dug_map_df.sources.notnull().astype(int)

ddg_map_df = pair_df.merge(disease_downregulates_gene_df, how='left')
ddg_map_df['hetionet'] = ddg_map_df.sources.notnull().astype(int)

print(dug_map_df.head(2))
print(ddg_map_df.head(2))

query = '''
SELECT "Disease_cid" AS doid_id, "Gene_cid" AS entrez_gene_id, count(*) AS n_sentences
FROM disease_gene
GROUP BY "Disease_cid", "Gene_cid";
'''
sentence_count_df = pd.read_sql(query, database_str)

dag_map_df = dag_map_df.merge(sentence_count_df, how='left')
dag_map_df.n_sentences = dag_map_df.n_sentences.fillna(0).astype(int)
dag_map_df['has_sentence'] = (dag_map_df.n_sentences > 0).astype(int)
dag_map_df.to_csv("disease_gene/disease_associates_gene/disease_associates_gene.tsv.xz", compression='xz', 
	sep="\t", index=False, float_format='%.5g')

dug_map_df = dug_map_df.merge(sentence_count_df, how='left')
dug_map_df.n_sentences = dug_map_df.n_sentences.fillna(0).astype(int)
dug_map_df['has_sentence'] = (dug_map_df.n_sentences > 0).astype(int)
dug_map_df.to_csv("../disease_gene/disease_upregulates_gene.tsv.xz", compression='xz', 
	sep="\t", index=False, float_format='%.5g')

ddg_map_df = ddg_map_df.merge(sentence_count_df, how='left')
ddg_map_df.n_sentences = ddg_map_df.n_sentences.fillna(0).astype(int)
ddg_map_df['has_sentence'] = (ddg_map_df.n_sentences > 0).astype(int)
ddg_map_df.to_csv("../disease_gene/disease_downregulates_gene.tsv.xz", compression='xz', 
	sep="\t", index=False, float_format='%.5g')