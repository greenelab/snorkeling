from snorkel.lf_helpers import (
    get_left_tokens,
    get_right_tokens,
    get_between_tokens,
    get_tagged_text,
    get_text_between,
    is_inverted,
    rule_regex_search_tagged_text,
    rule_regex_search_btw_AB,
    rule_regex_search_btw_BA,
    rule_regex_search_before_A,
    rule_regex_search_before_B,
)
from collections import OrderedDict
import numpy as np
import random
import re
import pathlib
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

random.seed(100)
stop_word_list = stopwords.words('english')

# Helper function for label functions
def ltp(tokens):
    return '(' + '|'.join(tokens) + ')'

"""
DISTANT SUPERVISION
"""
path = pathlib.Path(__file__).joinpath('../../../datafile/results/gene_interacts_gene.tsv.xz').resolve()
pair_df = pd.read_table(path, dtype={"sources": str})
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        key = str(row.gene1_id), str(row.gene2_id), source.lower()
        knowledge_base.add(key)

# Human Interactome Datasets
def LF_HETNET_HI_I_05(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'hi-i-05') in knowledge_base or 
        (c.Gene2_cid, c.Gene1_cid, 'hi-i-05')in knowledge_base 
        else 0
    )

def LF_HETNET_VENKATESAN_09(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'venkatesan-09') in knowledge_base or 
        (c.Gene2_cid, c.Gene1_cid, 'venkatesan-09') in knowledge_base
        else 0
    )

def LF_HETNET_YU_11(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'yu-11') in knowledge_base
        or (c.Gene2_cid, c.Gene1_cid, 'yu-11') in knowledge_base
        else 0
    )

def LF_HETNET_HI_II_14(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'hi-ii-14') in knowledge_base
        or (c.Gene2_cid, c.Gene1_cid, 'hi-ii-14') in knowledge_base 
        else 0
    )

def LF_HETNET_LIT_BM_13(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'lit-bm-13') in knowledge_base 
        or (c.Gene2_cid, c.Gene1_cid, 'lit-bm-13') in knowledge_base 
        else 0
    )

# Incomplete Interactome
def LF_HETNET_II_BINARY(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'ii-binary') in knowledge_base
        or (c.Gene2_cid, c.Gene1_cid, 'ii-binary') in knowledge_base
        else 0
    )

def LF_HETNET_II_LITERATURE(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'ii-literature') in knowledge_base 
        or (c.Gene2_cid, c.Gene1_cid, 'ii-literature') in knowledge_base 
        else 0
    )
# Hetionet
def LF_HETNET_HETIO_DAG(c):
    return (
        1 if (c.Gene1_cid, c.Gene2_cid, 'hetio-dag') in knowledge_base
        or (c.Gene2_cid, c.Gene1_cid, 'hetio-dag') in knowledge_base
        else 0
    )

def LF_HETNET_GiG_ABSENT(c):
    return 0 if any([
        LF_HETNET_HI_I_05(c),
        LF_HETNET_VENKATESAN_09(c),
        LF_HETNET_YU_11(c),
        LF_HETNET_HI_II_14(c),
        LF_HETNET_LIT_BM_13(c),
        LF_HETNET_II_BINARY(c),
        LF_HETNET_II_LITERATURE(c),
        LF_HETNET_HETIO_DAG(c)
    ]) else -1

"""
SENTENCE PATTERN MATCHING
"""
binding_identifiers =  {
    "interact(s with|ed)", "bind(s|ing)?",
    "phosphorylat(es|ion)", "heterodimeriz(e|ation)",
    "component binding", "multiple ligand binding",
    "cross-link(ing|ed)?", "mediates", "potential target for",
    "interaction( of|s with)", "receptor binding", "reactions",
    "phosphorylation by", "up-regulated by",
    "coactivators", "bound to"
}

cell_indications = {
    "cell(s)?", "\+", "-", "immunophenotyping",
    "surface marker analysis"
}

compound_indications = {
    "inhibitors", "therapy"
}

upregulates_identifiers = {
    "elevated( serum)?", "amplification of",
    "enhance(s|d)", "phsophorylation", "transcriptional activation",
    "potentiated", "stimulate production", "up-regulated"
}

downregulates_identifiers = {
    "decreased", "depressed", "inhibitory action",
    "competitive inhibition", "defective", "inihibit(ed|s)",
    "abrogated"
}

regulation_identifiers = {
    "mediates", "modulates", "stimulate production"
}

association_identifiers = {
    "associate(s|d)( with)?", "statsitically significant"
}

bound_identifiers = {
    "heterodimer(s)?", "receptor(s)?", "enzyme",
    "binding protein", "mediator"
}

gene_identifiers = {
    "variant(s)?", "markers", "gene",
    "antigen", "mutations( in| of)"
}

gene_adjective = {
    "responsive", "mediated"
}

diagnosis_indication = {
    "diagnostic markers", "diagnosis of"
}

method_indication = {
    "was determined", "was assayed",
    "removal of", "to assess", "the effect of",
    "was studied", "coeluted with", "we evaluated"
}

def LF_GiG_BINDING_IDENTIFICATIONS(c):
    gene1_tokens = list(get_left_tokens(c[0], window=5)) + list(get_right_tokens(c[0], window=5))
    gene2_tokens = list(get_left_tokens(c[0], window=5)) + list(get_right_tokens(c[0], window=5))

    if re.search(ltp(binding_identifiers), " ".join(gene1_tokens), flags=re.I):
        return 1
    elif re.search(ltp(binding_identifiers), " ".join(gene2_tokens), flags=re.I):
        return 1
    elif re.search(ltp(binding_identifiers), get_text_between(c), flags=re.I):
        return 1
    else:
        return 0

def LF_GiG_CELL_IDENTIFICATIONS(c):
    gene1_tokens = list(get_left_tokens(c[0], window=5)) + list(get_right_tokens(c[0], window=5))
    gene2_tokens = list(get_left_tokens(c[0], window=5)) + list(get_right_tokens(c[0], window=5))

    if re.search(ltp(cell_indications), " ".join(gene1_tokens), flags=re.I):
        return -1
    elif re.search(ltp(cell_indications), " ".join(gene2_tokens), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_COMPOUND_IDENTIFICATIONS(c):
    if re.search(ltp(compound_indications), " ".join(get_right_tokens(c[0], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(compound_indications), " ".join(get_right_tokens(c[1], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(compound_indications), get_text_between(c), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_UPREGULATES(c):
    if re.search(ltp(upregulates_identifiers), " ".join(get_right_tokens(c[0], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(upregulates_identifiers), " ".join(get_right_tokens(c[1], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(upregulates_identifiers), get_text_between(c), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_DOWNREGULATES(c):
    if re.search(ltp(downregulates_identifiers), " ".join(get_right_tokens(c[0], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(downregulates_identifiers), " ".join(get_right_tokens(c[1], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(downregulates_identifiers), get_text_between(c), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_REGULATION(c):
    if LF_GiG_UPREGULATES(c) or LF_GiG_DOWNREGULATES(c):
        return -1
    elif re.search(ltp(regulation_identifiers), get_text_between(c), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_ASSOCIATION(c):
    if re.search(ltp(association_identifiers), " ".join(get_right_tokens(c[0], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(association_identifiers), " ".join(get_right_tokens(c[1], window=2)), flags=re.I):
        return -1
    elif re.search(ltp(association_identifiers), get_text_between(c), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_BOUND_IDENTIFIERS(c):
    cand1_text = " ".join(list(get_left_tokens(c[0], window=5)) + list(get_right_tokens(c[0], window=5)))
    cand2_text = " ".join(list(get_left_tokens(c[1], window=5)) + list(get_right_tokens(c[1], window=5)))

    if re.search(ltp(bound_identifiers), cand1_text, flags=re.I):
        return 1
    elif re.search(ltp(bound_identifiers), cand2_text, flags=re.I):
        return 1
    else:
        return 0

def LF_GiG_GENE_IDENTIFIERS(c):
    cand1_text = " ".join(list(get_left_tokens(c[0], window=5)) + list(get_right_tokens(c[0], window=5)))
    cand2_text = " ".join(list(get_left_tokens(c[1], window=5)) + list(get_right_tokens(c[1], window=5)))

    if re.search(ltp(gene_identifiers), cand1_text, flags=re.I):
        return 1
    elif re.search(ltp(gene_identifiers), cand2_text, flags=re.I):
        return 1
    else:
        return 0

def LF_GiG_GENE_ADJECTIVE(c):
    if "-" in c[0].get_span() and re.search(ltp(gene_adjective), c[0].get_span(), flags=re.I):
        return -1
    elif "-" in c[1].get_span() and re.search(ltp(gene_adjective), c[1].get_span(), flags=re.I):
        return -1
    return 0

def LF_GiG_DIAGNOSIS_IDENTIFIERS(c):
    if re.search(ltp(diagnosis_indication), get_text_between(c), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_METHOD_DESC(c):
    sentence_tokens = " ".join(c.get_parent().words[0:20])
    if re.search(ltp(method_indication), sentence_tokens, flags=re.I):
        return -1
    elif re.search(ltp(method_indication), " ".join(get_between_tokens(c)), flags=re.I):
        return -1
    else:
        return 0

def LF_GiG_PARENTHESIS(c):
    if ")" in c[0].get_span() and LF_GG_DISTANCE_SHORT(c):
        return -1
    elif ")" in c[1].get_span() and LF_GG_DISTANCE_SHORT(c):
        return -1
    return 0

def LF_GG_IN_SERIES(c):
    if len(re.findall(r',', get_tagged_text(c))) >= 2:
        if re.search(', and', get_tagged_text(c)):
            return -1
    return 0

def LF_GiG_NO_CONCLUSION(c):
    positive_num = np.sum([
        LF_GiG_BINDING_IDENTIFICATIONS(c), LF_GiG_GENE_IDENTIFIERS(c),
        LF_GiG_BOUND_IDENTIFIERS(c), np.abs(LF_GiG_UPREGULATES(c)), 
        np.abs(LF_GiG_DOWNREGULATES(c)) 
    ])
    negative_num = np.abs(np.sum([
        LF_GiG_CELL_IDENTIFICATIONS(c), LF_GiG_COMPOUND_IDENTIFICATIONS(c), 
        LF_GG_NO_VERB(c), LF_GiG_PARENTHESIS(c), LF_GiG_DIAGNOSIS_IDENTIFIERS(c),

    ]))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_GiG_CONCLUSION(c):
    if not LF_GiG_NO_CONCLUSION(c):

        conclusion_sum = np.sum([
            LF_GiG_BINDING_IDENTIFICATIONS(c), LF_GiG_GENE_IDENTIFIERS(c),
            LF_GiG_BOUND_IDENTIFIERS(c), LF_GiG_UPREGULATES(c), LF_GiG_DOWNREGULATES(c)
        ])

        if conclusion_sum > 0:
            return 1
        else:
            return -1
    return 0

def LF_GG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't right next to each other.
    """
    return -1 if len(list(get_between_tokens(c))) <= 1 else 0

def LF_GG_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't too far from each other.
    """
    return -1 if len(list(get_between_tokens(c))) > 25 else 0

def LF_GG_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention are in an acceptable distance between 
    each other
    """
    return 0 if any([
        LF_GG_DISTANCE_LONG(c),
        LF_GG_DISTANCE_SHORT(c)
        ]) else 1

def LF_GG_NO_VERB(c):
    if len([x for x in  c.get_parent().pos_tags if "VB" in x and x != "VBG"]) == 0:
        return -1
    return 0

"""
Bi-Clustering LFs
"""
path = pathlib.Path(__file__).joinpath("../../../../../dependency_cluster/gene_gene_bicluster_results.tsv.xz").resolve()
bicluster_dep_df = pd.read_table(path)
binding_base = set([tuple(x) for x in bicluster_dep_df.query("B>0")[["pubmed_id", "sentence_num"]].values])
enhances_base = set([tuple(x) for x in bicluster_dep_df.query("W>0")[["pubmed_id", "sentence_num"]].values])
activates_base = set([tuple(x) for x in bicluster_dep_df[bicluster_dep_df["V+"]>0][["pubmed_id", "sentence_num"]].values])
increases_expression_base = set([tuple(x) for x in bicluster_dep_df[bicluster_dep_df["E+"]>0][["pubmed_id", "sentence_num"]].values])
affects_expression_base = set([tuple(x) for x in bicluster_dep_df.query("E>0")[["pubmed_id", "sentence_num"]].values])
signaling_base = set([tuple(x) for x in bicluster_dep_df.query("I>0")[["pubmed_id", "sentence_num"]].values])
identical_protein_base = set([tuple(x) for x in bicluster_dep_df.query("H>0")[["pubmed_id", "sentence_num"]].values])
regulation_base = set([tuple(x) for x in bicluster_dep_df.query("Rg>0")[["pubmed_id", "sentence_num"]].values])
cell_production_base = set([tuple(x) for x in bicluster_dep_df.query("Q>0")[["pubmed_id", "sentence_num"]].values])

def LF_GG_BICLUSTER_BINDING(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in binding_base:
        return 1
    return 0

def LF_GG_BICLUSTER_ENHANCES(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in enhances_base:
        return 1
    return 0

def LF_GG_BICLUSTER_ACTIVATES(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in activates_base:
        return 1
    return 0

def LF_GG_BICLUSTER_INCREASES_EXPRESSION(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in increases_expression_base:
        return 1
    return 0

def LF_GG_BICLUSTER_AFFECTS_EXPRESSION(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in affects_expression_base:
        return 1
    return 0

def LF_GG_BICLUSTER_SIGNALING(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in signaling_base:
        return 1
    return 0

def LF_GG_BICLUSTER_IDENTICAL_PROTEIN(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in identical_protein_base:
        return 1
    return 0

def LF_GG_BICLUSTER_REGULATION(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in regulation_base:
        return 1
    return 0

def LF_GG_BICLUSTER_CELL_PRODUCTION(c):
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in cell_production_base:
        return 1
    return 0


GG_LFS = {
    "GiG":OrderedDict({
        "LF_HETNET_HI_I_05":LF_HETNET_HI_I_05,
        "LF_HETNET_VENKATESAN_09":LF_HETNET_VENKATESAN_09,
        "LF_HETNET_YU_11":LF_HETNET_YU_11,
        "LF_HETNET_HI_II_14":LF_HETNET_HI_II_14,
        "LF_HETNET_LIT_BM_13":LF_HETNET_LIT_BM_13,
        "LF_HETNET_II_BINARY":LF_HETNET_II_BINARY,
        "LF_HETNET_II_LITERATURE":LF_HETNET_II_LITERATURE,
        "LF_HETNET_HETIO_DAG":LF_HETNET_HETIO_DAG,
        "LF_HETNET_GiG_ABSENT":LF_HETNET_GiG_ABSENT,
        "LF_GiG_BINDING_IDENTIFICATIONS":LF_GiG_BINDING_IDENTIFICATIONS,
        "LF_GiG_CELL_IDENTIFICATIONS":LF_GiG_CELL_IDENTIFICATIONS,
        "LF_GiG_COMPOUND_IDENTIFICATIONS":LF_GiG_COMPOUND_IDENTIFICATIONS,
        "LF_GiG_UPREGULATES":LF_GiG_UPREGULATES,
        "LF_GiG_DOWNREGULATES":LF_GiG_DOWNREGULATES,
        "LF_GiG_REGULATION":LF_GiG_REGULATION,
        "LF_GiG_ASSOCIATION":LF_GiG_ASSOCIATION,
        "LF_GiG_BOUND_IDENTIFIERS":LF_GiG_BOUND_IDENTIFIERS,
        "LF_GiG_GENE_IDENTIFIERS":LF_GiG_GENE_IDENTIFIERS,
        "LF_GiG_GENE_ADJECTIVE":LF_GiG_GENE_ADJECTIVE,
        "LF_GiG_DIAGNOSIS_IDENTIFIERS":LF_GiG_DIAGNOSIS_IDENTIFIERS,
        "LF_GiG_METHOD_DESC":LF_GiG_METHOD_DESC,
        "LF_GiG_PARENTHESIS":LF_GiG_PARENTHESIS,
        "LF_GG_IN_SERIES":LF_GG_IN_SERIES,
        "LF_GiG_NO_CONCLUSION":LF_GiG_NO_CONCLUSION,
        "LF_GiG_CONCLUSION":LF_GiG_CONCLUSION,
        "LF_GG_DISTANCE_SHORT":LF_GG_DISTANCE_SHORT,
        "LF_GG_DISTANCE_LONG":LF_GG_DISTANCE_LONG,
        "LF_GG_ALLOWED_DISTANCE":LF_GG_ALLOWED_DISTANCE,
        "LF_GG_NO_VERB":LF_GG_NO_VERB,
        "LF_GG_BICLUSTER_BINDING":LF_GG_BICLUSTER_BINDING,
        "LF_GG_BICLUSTER_ENHANCES":LF_GG_BICLUSTER_ENHANCES,
        "LF_GG_BICLUSTER_ACTIVATES":LF_GG_BICLUSTER_ACTIVATES,
        "LF_GG_BICLUSTER_AFFECTS_EXPRESSION":LF_GG_BICLUSTER_AFFECTS_EXPRESSION,
        "LF_GG_BICLUSTER_INCREASES_EXPRESSION":lambda x: -1*LF_GG_BICLUSTER_INCREASES_EXPRESSION(x),
        "LF_GG_BICLUSTER_SIGNALING":LF_GG_BICLUSTER_SIGNALING,
        "LF_GG_BICLUSTER_IDENTICAL_PROTEIN":lambda x: -1*LF_GG_BICLUSTER_IDENTICAL_PROTEIN(x),
        "LF_GG_BICLUSTER_CELL_PRODUCTION":lambda x: -1*LF_GG_BICLUSTER_CELL_PRODUCTION(x),
    }),
    "GrG":OrderedDict({
        "LF_GG_BICLUSTER_ENHANCES":LF_GG_BICLUSTER_ENHANCES,
        "LF_GG_BICLUSTER_AFFECTS_EXPRESSION":LF_GG_BICLUSTER_AFFECTS_EXPRESSION,
        "LF_GG_BICLUSTER_INCREASES_EXPRESSION":LF_GG_BICLUSTER_INCREASES_EXPRESSION,
        "LF_GG_BICLUSTER_SIGNALING":LF_GG_BICLUSTER_SIGNALING,
        "LF_GG_BICLUSTER_REGULATION":LF_GG_BICLUSTER_REGULATION,
        "LF_GG_BICLUSTER_IDENTICAL_PROTEIN":lambda x: -1*LF_GG_BICLUSTER_IDENTICAL_PROTEIN(x),
        "LF_GG_BICLUSTER_CELL_PRODUCTION":lambda x: -1*LF_GG_BICLUSTER_CELL_PRODUCTION(x),
    })
}