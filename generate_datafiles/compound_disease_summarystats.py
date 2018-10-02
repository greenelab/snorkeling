from collections import defaultdict
import itertools
import statistics
import pandas as pd

import tqdm

from hetio.permute import permute_pair_list

hetnet_df = pd.read_csv("compound_degree/compound_treats_disease/compound_treats_disease.csv.xz", compression='xz')
disease_degree = dict(hetnet_df["doid_id"].value_counts())
compound_degree = dict(hetnet_df["drugbank_id"].value_counts())


treats_edge = defaultdict(set)
treats_row = list()

for (disease, d_degree), (compound, c_degree) in tqdm.tqdm(itertools.product(disease_degree.items(), compound_degree.items())):
    treats_row.append((disease, compound, d_degree, c_degree))
    treats_edge[(d_degree, c_degree)].add((disease, compound))

pair_df = pd.DataFrame(treats_row, columns=["disease_id", "compound_id", "disease_treats", "compound_treats"])

associations = list(zip(hetnet_df["disease_id"], hetnet_df["compound_id"]))

pair_list, stats = permute_pair_list(associations, multiplier=10)
burnin_stats = pd.DataFrame(stats)

multiplier = 3

# calculate the total number of permutations
# divide the total number by half to prevent memory issues
n_perm = hetnet_df["disease_id"].nunique() * hetnet_df["compound_id"].nunique()
n_perm = int(n_perm * 0.5)

edges_to_prob = {x: list() for x in treats_edge}

for i in tqdm.tqdm(range(n_perm)):
    pair_list, stats = permute_pair_list(pair_list, multiplier=multiplier, seed=i)
    
    pair_set = set(pair_list)
    for degree, probs in edges_to_prob.items():
        edges = treats_edge[degree]
        probs.append(len(edges & pair_set) / len(edges))

rows = []

for (d_deg, c_deg), probs in tqdm.tqdm(edges_to_prob.items()):
    mean = statistics.mean(probs)
    std_error = statistics.stdev(probs) / len(probs) ** 0.5
    rows.append((d_deg, c_deg, mean, std_error))
    
perm_df = pd.DataFrame(rows, columns=['disease_treats', 'compound_treats', 'prior_perm', 'prior_perm_stderr'])


# Add unpermuted treatment prevalence columns
rows = list()
association_set = set(associations)

for (d_deg, c_deg), edges in treats_edge.items():
    n_treats = len(edges & association_set)
    rows.append((d_deg, c_deg, n_treats, len(edges)))

degree_prior_df = pd.DataFrame(rows, columns=['disease_treats', 'compound_treats', 'n_treats', 'n_possible'])
degree_prior_df = perm_df.merge(degree_prior_df)
degree_prior_df = degree_prior_df.sort_values(['disease_treats', 'compound_treats'], ascending=False)

obs_pair_df = pair_df.merge(perm_df)

degree_prior_df.to_csv("../compound_degree/compound_treats_disease/degree-prior.tsv", sep="\t", index=False, float_format='%.6g')
obs_pair_df.to_csv("../compound_degree/compound_treats_disease/observation-prior.tsv", sep="\t", index=False, float_format='%.6g')