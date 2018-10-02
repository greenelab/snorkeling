from collections import defaultdict
import itertools
import statistics
import pandas as pd

from hetio.permute import permute_pair_list

hetnet_df = pd.read_csv("diseae_gene/disease_associates_gene/disease_gene_pairs_association.csv.xz", compression='xz')
disease_degree = dict(hetnet_df["doid_id"].value_counts())
gene_degree = dict(hetnet_df["entrez_gene_id"].value_counts())


association_edge = defaultdict(set)
association_row = list()

for (disease, d_degree), (gene, g_degree) in tqdm.tqdm(itertools.product(disease_degree.items(), gene_degree.items())):
    association_row.append((disease, gene, d_degree, g_degree))
    association_edge[(d_degree, g_degree)].add((disease, gene))

pair_df = pd.DataFrame(association_row, columns=["disease_id", "gene_id", "disease_associates", "gene_associates"])

associations = list(zip(hetnet_df["disease_id"], hetnet_df["gene_id"]))

pair_list, stats = permute_pair_list(associations, multiplier=10)
burnin_stats = pd.DataFrame(stats)

multiplier = 3

# calculate the total number of permutations
# divide the total number by half to prevent memory issues
n_perm = hetnet_df["disease_id"].nunique() * hetnet_df["gene_id"].nunique()
n_perm = int(n_perm * 0.5)

edges_to_prob = {x: list() for x in association_edge}

for i in tqdm.tqdm(range(n_perm)):
    pair_list, stats = permute_pair_list(pair_list, multiplier=multiplier, seed=i)
    
    pair_set = set(pair_list)
    for degree, probs in edges_to_prob.items():
        edges = association_edge[degree]
        probs.append(len(edges & pair_set) / len(edges))

rows = []

for (d_deg, g_deg), probs in tqdm.tqdm(edges_to_prob.items()):
    mean = statistics.mean(probs)
    std_error = statistics.stdev(probs) / len(probs) ** 0.5
    rows.append((d_deg, g_deg, mean, std_error))
    
perm_df = pd.DataFrame(rows, columns=['disease_associates', 'gene_associates', 'prior_perm', 'prior_perm_stderr'])


# Add unpermuted treatment prevalence columns
rows = list()
association_set = set(associations)

for (d_deg, g_deg), edges in association_edge.items():
    n_associations = len(edges & association_set)
    rows.append((d_deg, g_deg, n_associations, len(edges)))

degree_prior_df = pd.DataFrame(rows, columns=['disease_associates', 'gene_associates', 'n_associations', 'n_possible'])
degree_prior_df = perm_df.merge(degree_prior_df)
degree_prior_df = degree_prior_df.sort_values(['disease_associates', 'gene_associates'], ascending=False)

obs_pair_df = pair_df.merge(perm_df)

degree_prior_df.to_csv("../disease_gene/disease_associates_gene/degree-prior.tsv", sep="\t", index=False, float_format='%.6g')
obs_pair_df.to_csv("../disease_gene/disease_associates_gene/observation-prior.tsv", sep="\t", index=False, float_format='%.6g')