
# coding: utf-8

# # Construct the epilepsy gene gold standard (knowledgebase)
# 
# See https://github.com/greenelab/snorkeling/issues/9

# In[1]:

import pandas
from sklearn.cross_validation import StratifiedShuffleSplit
from neo4j.v1 import GraphDatabase


# In[2]:

driver = GraphDatabase.driver("bolt://neo4j.het.io")


# In[3]:

query = '''MATCH (gene:Gene)
OPTIONAL MATCH (gene)-[assoc:ASSOCIATES_DaG]-(disease:Disease)
WHERE disease.name = 'epilepsy syndrome'
RETURN
 gene.identifier AS entrez_gene_id,
 gene.name AS gene_symbol,
 gene.description AS gene_name,
 count(assoc) AS positive,
 assoc.sources AS sources
ORDER BY entrez_gene_id
'''


# In[4]:

with driver.session() as session:
    result = session.run(query)
    assoc_df = pandas.DataFrame((x.values() for x in result), columns=result.keys())
len(assoc_df)


# In[5]:

# Comma separate sources
assoc_df.sources = assoc_df.sources.str.join(', ')


# In[6]:

assoc_df.positive.value_counts()


# In[7]:

# Assign testing observations
(train, test), = StratifiedShuffleSplit(assoc_df.positive, n_iter=1, test_size=0.3, random_state=0)
assoc_df['testing'] = 0
assoc_df.loc[test, 'testing'] = 1
assoc_df.testing.value_counts()


# In[8]:

# Breakdown of gene assignments
pandas.crosstab(assoc_df.positive, assoc_df.testing)


# In[9]:

assoc_df.head(3)


# In[10]:

assoc_df.to_csv('epilepsy-genes.tsv', sep='\t', index=False)

