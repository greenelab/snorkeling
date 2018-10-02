from collections import defaultdict
import itertools
import statistics
import pandas as pd

from hetio.permute import permute_pair_list

hetnet_df = pd.read_csv("compound_gene/compound_binds_gene/compound_gene_pairs_binds.csv")
compound_degree = dict(hetnet_df["drugbank_id"].value_counts())
gene_degree = dict(hetnet_df["entrez_gene_id"].value_counts())


binds_edge = defaultdict(set)
binds_row = list()

for (compund, c_degree), (gene, g_degree) in tqdm.tqdm(itertools.product(disease_degree.items(), gene_degree.items())):
    binds_row.append((compound, gene, c_degree, g_degree))
    association_edge[(c_degree, g_degree)].add((compound, gene))

pair_df = pd.DataFrame(binds_row, columns=["compound_id", "gene_id", "compound_binds", "gene_binds"])

binds = list(zip(hetnet_df["compound_id"], hetnet_df["gene_id"]))

pair_list, stats = permute_pair_list(binds, multiplier=10)
burnin_stats = pd.DataFrame(stats)

multiplier = 3

# calculate the total number of permutations
# divide the total number by half to prevent memory issues
n_perm = hetnet_df["compound_id"].nunique() * hetnet_df["gene_id"].nunique()
n_perm = int(n_perm * 0.5)

edges_to_prob = {x: list() for x in binds_edge}

for i in tqdm.tqdm(range(n_perm)):
    pair_list, stats = permute_pair_list(pair_list, multiplier=multiplier, seed=i)
    
    pair_set = set(pair_list)
    for degree, probs in edges_to_prob.items():
        edges = binds_edge[degree]
        probs.append(len(edges & pair_set) / len(edges))

rows = []

for (d_deg, g_deg), probs in tqdm.tqdm(edges_to_prob.items()):
    mean = statistics.mean(probs)
    std_error = statistics.stdev(probs) / len(probs) ** 0.5
    rows.append((d_deg, g_deg, mean, std_error))
    
perm_df = pd.DataFrame(rows, columns=['compound_binds', 'gene_binds', 'prior_perm', 'prior_perm_stderr'])


# Add unpermuted treatment prevalence columns
rows = list()
binds_set = set(binds)

for (c_deg, g_deg), edges in association_edge.items():
    n_associations = len(edges & binds_set)
    rows.append((c_deg, g_deg, n_binds, len(edges)))

degree_prior_df = pd.DataFrame(rows, columns=['compound_binds', 'gene_binds', 'n_binds', 'n_possible'])
degree_prior_df = perm_df.merge(degree_prior_df)
degree_prior_df = degree_prior_df.sort_values(['compound_binds', 'gene_binds'], ascending=False)

obs_pair_df = pair_df.merge(perm_df)

degree_prior_df.to_csv("../compound_gene/compound_binds_gene/degree-prior.tsv", sep="\t", index=False, float_format='%.6g')
obs_pair_df.to_csv("../compound_gene/compound_binds_gene/observation-prior.tsv", sep="\t", index=False, float_format='%.6g')