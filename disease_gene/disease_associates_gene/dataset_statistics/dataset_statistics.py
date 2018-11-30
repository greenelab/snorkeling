
# coding: utf-8

# # Dataset Statistics for Disease Gene Sentences

# This notebook is designed to show statistics on the data extracted from pubmed. The following cells below here are needed to set up the environment.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter
from itertools import product
import os
import pickle
import sys

sys.path.append(os.path.abspath('../../../modules'))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import seaborn as sns
from tqdm import tqdm_notebook


# In[2]:


#Set up the environment
username = "danich1"
password = "snorkel"
dbname = "pubmeddb"

#Path subject to change for different os
database_str = "postgresql+psycopg2://{}:{}@/{}?host=/var/run/postgresql".format(username, password, dbname)
os.environ['SNORKELDB'] = database_str

from snorkel import SnorkelSession
session = SnorkelSession()


# In[3]:


from snorkel.models import candidate_subclass, Candidate
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])


# In[4]:


from utils.notebook_utils.dataframe_helper import write_candidates_to_excel


# # Read Full Sentence Table

# The cells below will read every sentence that contains a gene and disease entity from the sentence table in our postgres database. For time sake majority of the data has already been processed and save as files mentioned below.

# In[ ]:


sql = '''
select id, text, entity_cids, entity_types 
from sentence 
where entity_types::text like '%%Gene%%' or entity_types::text like '%%Disease%%';
'''
sentence_df = pd.read_sql(sql, database_str)
sentence_df.head(2)


# In[ ]:


records = []
for index, row in tqdm_notebook(sentence_df.iterrows()):
    entity_record = Counter([x for x in row['entity_types'] if x !='O'])
    entity_record.update({'sentence_id':row['id']})
    records.append(dict(entity_record))
sentence_stats_df = pd.DataFrame.from_records(records).fillna(0)
sentence_stats_df.to_csv("data/sentence_stats.tsv.xz", sep="\t",  index=False)


# In[ ]:


entity_counts = Counter()
for index, row in tqdm_notebook(sentence_df.iterrows()):
    entity_counts.update([x for x in row['entity_cids'] if x!='O'])
sentence_stats_df = pd.DataFrame.from_dict(records).fillna(0)
sentence_stats_df.to_csv("data/entity_stats.tsv.xz", sep="\t",  index=False)


# # Stratified Sentence Counts and Statistics

# Below are the counts of sentences that contain a co-occuring pair in and not in hetionet. Sentences can contain multiple co=ccuring pairs, so these results do have single sentences counted multiple times.

# In[5]:


entity_level_df = pd.read_csv("../disease_gene_pairs_association.csv.xz")
entity_level_df.head(2)


# In[6]:


sentence_stats_df = pd.read_table("data/sentence_stats.tsv.xz")


# In[7]:


sentence_sql = '''
select cand_id, "Disease_cid", "Gene_cid", sentence_id from (
select disease_gene.id as "cand_id", disease_gene."Disease_id",  disease_gene."Disease_cid", disease_gene."Gene_cid", candidate.split from disease_gene 
inner join candidate on disease_gene.id=candidate.id 
where split={}
) as candidate_splits inner join span on candidate_splits."Disease_id"=span.id;
'''
train_candidate_df = pd.read_sql(sentence_sql.format(0), database_str)
dev_candidate_df = pd.read_sql(sentence_sql.format(1), database_str)
test_candidate_df = pd.read_sql(sentence_sql.format(2), database_str)


# In[8]:


clean_up_df = lambda x: (
        sentence_stats_df
        .merge(x, on="sentence_id")
        .rename(index=str, columns={
            "Disease":"Disease_mention_count", 
            "Gene":"Gene_mention_count", 
            "Compound":"Compound_mention_count",
            "cand_id": "candidate_id",
            "Disease_cid": "doid_id",
            "Gene_cid": "entrez_gene_id"
        })
        .astype({"entrez_gene_id": int})
    )


# In[9]:


train_candidate_df = clean_up_df(train_candidate_df)
dev_candidate_df = clean_up_df(dev_candidate_df)
test_candidate_df = clean_up_df(test_candidate_df)


# In[10]:


train_candidate_df.head(2)


# In[12]:


training_set_df = (
    entity_level_df
    .query("split==0&has_sentence==1")
    .merge(
        train_candidate_df, 
        on=["entrez_gene_id", "doid_id"]
    )
)

dev_set_df = (
    entity_level_df
    .query("split==1&has_sentence==1")
    .merge(
        dev_candidate_df, 
        on=["entrez_gene_id", "doid_id"]
    )
)

test_set_df = (
    entity_level_df
    .query("split==2&has_sentence==1")
    .merge(
        test_candidate_df, 
        on=["entrez_gene_id", "doid_id"]
    )
)


# In[13]:


total_candidates_df = (
    training_set_df
    .append(dev_set_df)
    .append(test_set_df)
)


# In[14]:


venn2(
    [
        set(total_candidates_df.query("hetionet==0").sentence_id),
        set(total_candidates_df.query("hetionet==1").sentence_id)
    ], set_labels=["Not In Hetionet", "In Hetionet"])
plt.title("# of Unique Sentences in Entire Dataset with Co-Mention Pair in/not in hetionet")


# In[15]:


venn2(
    [
        set(training_set_df.query("hetionet==0").sentence_id),
        set(training_set_df.query("hetionet==1").sentence_id)
    ], set_labels=["Not in Hetionet", "In Hetionet"])
plt.title("# of Unique Sentences in Training Set with Co-mentions in/not in Hetionet")
plt.show()


# In[16]:


venn2(
    [
        set(dev_set_df.query("hetionet==0").sentence_id),
        set(dev_set_df.query("hetionet==1").sentence_id)
    ], set_labels=["Not in Hetionet", "In Hetionet"])
plt.title("# of Unique Sentences in Dev Set with Co-mentions in/not in Hetionet")
plt.show()


# In[17]:


venn2(
    [
        set(test_set_df.query("hetionet==0").sentence_id),
        set(test_set_df.query("hetionet==1").sentence_id)
    ], set_labels=["Not in Hetionet", "In Hetionet"])
plt.title("# of Unique Sentences in Test Set with Co-mentions in/not in Hetionet")
plt.show()


# # Co-occuring Mentions Sentence Stats

# This next block contains visualizations for the distribution of mentions occuring in a given sentence. These visualizations are heatmaps that represent the counts of sentences that fall into certain mention groups

# In[18]:


sns.set(rc={'figure.figsize':(10,6), "font.size":17})
ax = sns.scatterplot(
    x="Disease_mention_count", 
    y="Gene_mention_count", 
    data=
    (
        total_candidates_df
        .drop_duplicates("sentence_id")
        .groupby(["Disease_mention_count", "Gene_mention_count"]).size()
        .reset_index()
        .rename(index=str, columns={0:"size"})
        .assign(natural_log_size= lambda x: pd.np.log(x['size']))
    ),
    hue="natural_log_size",
    palette='viridis'
)
ax.set_title("Sentence Distribution of Entire Dataset")


# In[19]:


sns.set(rc={'figure.figsize':(10,6)})
ax = sns.scatterplot(
    x="Disease_mention_count", 
    y="Gene_mention_count", 
    data=
    (
        train_candidate_df
        .drop_duplicates("sentence_id")
        .groupby(["Disease_mention_count", "Gene_mention_count"]).size()
        .reset_index()
        .rename(index=str, columns={0:"size"})
        .assign(natural_log_size= lambda x: pd.np.log(x['size']))
    ),
    hue="natural_log_size",
    palette='viridis'
)
ax.set_title("Sentence Distribution of Training Dataset")


# Notice that there are a few outliers, especially one that contains a total of 175 gene mentions in one sentence. Upon closer inspection this sentence is actually a listing of abbreviations that could be removed to filter out a bit of noise.

# In[20]:


sns.set(rc={'figure.figsize':(10,6)})
ax = sns.scatterplot(
    x="Disease_mention_count", 
    y="Gene_mention_count", 
    data=
    (
        dev_candidate_df
        .drop_duplicates("sentence_id")
        .groupby(["Disease_mention_count", "Gene_mention_count"]).size()
        .reset_index()
        .rename(index=str, columns={0:"size"})
        .assign(natural_log_size= lambda x: pd.np.log(x['size']))
    ),
    hue="natural_log_size",
    palette='viridis'
)
ax.set_title("Sentence Distribution of Dev Dataset")


# In[21]:


sns.set(rc={'figure.figsize':(10,6)})
ax = sns.scatterplot(
    x="Disease_mention_count", 
    y="Gene_mention_count", 
    data=
    (
        test_candidate_df
        .drop_duplicates("sentence_id")
        .groupby(["Disease_mention_count", "Gene_mention_count"]).size()
        .reset_index()
        .rename(index=str, columns={0:"size"})
        .assign(natural_log_size= lambda x: pd.np.log(x['size']))
    ),
    hue="natural_log_size",
    palette='viridis'
)
ax.set_title("Sentence Distribution of Test Dataset")


# In[ ]:


candidate_files = zip(
    [train_candidate_df, dev_candidate_df, test_candidate_df], 
    ["data/train_sentence_stats.tsv.xz", "data/dev_sentence_stats.tsv.xz", "data/test_sentence_stats.tsv.xz"]
)

for dataframe, file in candidate_files:
    dataframe.to_csv(file, sep="\t", index=False, compression='xz')


# # Subsampled Set Distributions

# In[22]:


train_dataset_df = pd.read_excel("sentence_labels_train.xlsx")
dev_dataset_df = pd.read_excel("sentence_labels_dev.xlsx").query("curated_dsh.notnull()")
test_dataset_df = pd.read_excel("sentence_labels_test.xlsx").query("curated_dsh.notnull()")


# In[23]:


train_candidates_stats_df = (
    train_candidate_df
    .merge(train_dataset_df[["candidate_id"]], on="candidate_id")
)

dev_candidates_stats_df = (
    dev_candidate_df
    .merge(dev_dataset_df[["candidate_id"]], on="candidate_id")
)

test_candidates_stats_df = (
    train_candidate_df
    .merge(test_dataset_df, on="candidate_id")
)


# In[24]:


sns.set(rc={'figure.figsize':(10,6)})
ax = sns.scatterplot(
    x="Disease_mention_count", 
    y="Gene_mention_count", 
    data=
    (
        train_candidates_stats_df
        .drop_duplicates("sentence_id")
        .groupby(["Disease_mention_count", "Gene_mention_count"]).size()
        .reset_index()
        .rename(index=str, columns={0:"size"})
        .assign(natural_log_size= lambda x: pd.np.log(x['size']))
    ),
    hue="natural_log_size",
    palette='viridis'
)
ax.set_title("Distribution of Sentences in Subsampled Training Set")


# In[25]:


sns.set(rc={'figure.figsize':(10,6), 'font.size':30})
ax = sns.scatterplot(
    x="Disease_mention_count", 
    y="Gene_mention_count", 
    data=
    (
        dev_candidates_stats_df
        .drop_duplicates("sentence_id")
        .groupby(["Disease_mention_count", "Gene_mention_count"]).size()
        .reset_index()
        .rename(index=str, columns={0:"size"})
        .assign(natural_log_size= lambda x: pd.np.log(x['size']))
    ),
    hue="natural_log_size",
    palette='viridis'
)
ax.set_title("Sentence Distribution of Hand Labeled Dev Dataset")


# In[26]:


dev_dataset_df.curated_dsh.value_counts()

