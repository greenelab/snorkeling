
# coding: utf-8

# # Biclustering of Dependency Paths for Biomedical Realtionship Extraction

# A global network of biomedical relationships derived from text

# In[1]:


import pandas as pd


# # Chemical-Disease

# In[2]:


chemical_disease_url = 'https://zenodo.org/record/1495808/files/part-i-chemical-disease-path-theme-distributions.txt.zip'
chemical_disease_paths_url = 'https://zenodo.org/record/1495808/files/part-ii-dependency-paths-chemical-disease-sorted-with-themes.txt.zip'


# In[3]:


chemical_disease_path_dist_df = pd.read_table(chemical_disease_url)
chemical_disease_path_dist_df.head(2)


# In[4]:


chemical_disease_paths_df = pd.read_table(
    chemical_disease_paths_url, 
    names=[
        "pubmed_id", "sentence_num",
        "first_entity_name", "first_entity_location",
        "second_entity_name", "second_entity_location",
        "first_entity_name_raw", "second_entity_name_raw",
        "first_entity_db_id", "second_entity_db_id",
        "first_entity_type", "second_entity_type",
        "dep_path", "sentence"
    ]
)
chemical_disease_paths_df.head(2)


# In[5]:


chemical_disease_merged_path_df=(
    chemical_disease_paths_df
    .assign(dep_path=chemical_disease_paths_df.dep_path.apply(lambda x: x.lower()).values)
    .merge(chemical_disease_path_dist_df.rename(index=str, columns={"path":"dep_path"}), on=["dep_path"])
)
chemical_disease_merged_path_df.head(2)


# In[6]:


chemical_disease_merged_path_df.to_csv(
    "chemical_disease_bicluster_results.tsv.xz", 
    sep="\t", index=False, compression="xz"
)


# # Chemical-Gene

# In[7]:


chemical_gene_url = 'https://zenodo.org/record/1495808/files/part-i-chemical-gene-path-theme-distributions.txt.zip'
chemical_gene_paths_url = 'https://zenodo.org/record/1495808/files/part-ii-dependency-paths-chemical-gene-sorted-with-themes.txt.zip'


# In[8]:


chemical_gene_path_dist_df = pd.read_table(chemical_gene_url)
chemical_gene_path_dist_df.head(2)


# In[9]:


chemical_gene_paths_df = pd.read_table(
    chemical_gene_paths_url, 
    names=[
        "pubmed_id", "sentence_num",
        "first_entity_name", "first_entity_location",
        "second_entity_name", "second_entity_location",
        "first_entity_name_raw", "second_entity_name_raw",
        "first_entity_db_id", "second_entity_db_id",
        "first_entity_type", "second_entity_type",
        "dep_path", "sentence"
    ]
)
chemical_gene_paths_df.head(2)


# In[10]:


chemical_gene_merged_path_df=(
    chemical_gene_paths_df
    .assign(dep_path=chemical_gene_paths_df.dep_path.apply(lambda x: x.lower()).values)
    .merge(chemical_gene_path_dist_df.rename(index=str, columns={"path":"dep_path"}), on=["dep_path"])
)
chemical_gene_merged_path_df.head(2)


# In[11]:


chemical_gene_merged_path_df.to_csv(
    "chemical_gene_bicluster_results.tsv.xz", 
    sep="\t", index=False, compression="xz"
)


# # Disease-Gene

# In[12]:


disease_gene_url = 'https://zenodo.org/record/1495808/files/part-i-gene-disease-path-theme-distributions.txt.zip'
disease_gene_paths_url = 'https://zenodo.org/record/1495808/files/part-ii-dependency-paths-gene-disease-sorted-with-themes.txt.zip'


# In[13]:


disease_gene_path_dist_df = pd.read_table(disease_gene_url)
disease_gene_path_dist_df.head(2)


# In[14]:


disease_gene_paths_df = pd.read_table(
    disease_gene_paths_url, 
    names=[
        "pubmed_id", "sentence_num",
        "first_entity_name", "first_entity_location",
        "second_entity_name", "second_entity_location",
        "first_entity_name_raw", "second_entity_name_raw",
        "first_entity_db_id", "second_entity_db_id",
        "first_entity_type", "second_entity_type",
        "dep_path", "sentence"
    ]
)
disease_gene_paths_df.head(2)


# In[15]:


disease_gene_merged_path_df=(
    disease_gene_paths_df
    .assign(dep_path=disease_gene_paths_df.dep_path.apply(lambda x: x.lower()).values)
    .merge(disease_gene_path_dist_df.rename(index=str, columns={"path":"dep_path"}), on=["dep_path"])
)
disease_gene_merged_path_df.head(2)


# In[16]:


disease_gene_merged_path_df.to_csv(
    "disease_gene_bicluster_results.tsv.xz", 
    sep="\t", index=False, compression="xz"
)


# # Gene-Gene

# In[17]:


gene_gene_url = 'https://zenodo.org/record/1495808/files/part-i-gene-gene-path-theme-distributions.txt.zip'
gene_gene_paths_url = 'https://zenodo.org/record/1495808/files/part-ii-dependency-paths-gene-gene-sorted-with-themes.txt.zip'


# In[18]:


gene_gene_path_dist_df = pd.read_table(gene_gene_url)
gene_gene_path_dist_df.head(2)


# In[19]:


gene_gene_paths_df = pd.read_table(
    gene_gene_paths_url, 
    names=[
        "pubmed_id", "sentence_num",
        "first_entity_name", "first_entity_location",
        "second_entity_name", "second_entity_location",
        "first_entity_name_raw", "second_entity_name_raw",
        "first_entity_db_id", "second_entity_db_id",
        "first_entity_type", "second_entity_type",
        "dep_path", "sentence"
    ]
)
gene_gene_paths_df.head(2)


# In[20]:


gene_gene_merged_path_df=(
    gene_gene_paths_df
    .assign(dep_path=gene_gene_paths_df.dep_path.apply(lambda x: x.lower()).values)
    .merge(gene_gene_path_dist_df.rename(index=str, columns={"path":"dep_path"}), on=["dep_path"])
)
gene_gene_merged_path_df.head(2)


# In[21]:


gene_gene_merged_path_df.to_csv(
    "gene_gene_bicluster_results.tsv.xz", 
    sep="\t", index=False, compression="xz"
)

