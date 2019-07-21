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

"""
Debugging to understand how LFs work
"""


def LF_DEBUG(c):
    """
    This label function is for debugging purposes. Feel free to ignore.
    keyword arguments:
    c - The candidate object to be labeled
    """
    print(c)
    print()
    print("Left Tokens")
    print(list(get_left_tokens(c[0], window=5)))
    print()
    print("Right Tokens")
    print(list(get_right_tokens(c[0])))
    print()
    print("Between Tokens")
    print(list(get_between_tokens(c)))
    print() 
    print("Tagged Text")
    print(get_tagged_text(c))
    print(re.search(r'{{B}} .* is a .* {{A}}', get_tagged_text(c)))
    print()
    print("Get between Text")
    print(get_text_between(c))
    print(len(get_text_between(c)))
    print()
    print("Parent Text")
    print(c.get_parent())
    print()
    return 0


# Helper function for label functions
def ltp(tokens):
    return '(' + '|'.join(tokens) + ')'


"""
DISTANT SUPERVISION
"""
path = pathlib.Path(__file__).joinpath('../../compound_gene_pairs_binds.csv').resolve()
pair_df = pd.read_csv(path, dtype={"sources": str})
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        source = re.sub(r' \(\w+\)', '', source)
        key = str(row.entrez_gene_id), row.drugbank_id, source
        knowledge_base.add(key)

def LF_HETNET_DRUGBANK(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the Drugbank database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "DrugBank") in knowledge_base else 0

def LF_HETNET_DRUGCENTRAL(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the Drugcentral database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "DrugCentral") in knowledge_base else 0

def LF_HETNET_ChEMBL(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the ChEMBL database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "ChEMBL") in knowledge_base else 0

def LF_HETNET_BINDINGDB(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the BindingDB database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "BindingDB") in knowledge_base else 0

def LF_HETNET_PDSP_KI(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the PDSP_KI database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "PDSP Ki") in knowledge_base else 0

def LF_HETNET_US_PATENT(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the US PATENT database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "US Patent") in knowledge_base else 0

def LF_HETNET_PUBCHEM(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the PUBCHEM database
    """
    return 1 if (c.Gene_cid, c.Compound_cid, "PubChem") in knowledge_base else 0

def LF_HETNET_CG_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear 
    in the databases above.
    """
    return 0 if any([
        LF_HETNET_DRUGBANK(c),
        LF_HETNET_DRUGCENTRAL(c),
        LF_HETNET_ChEMBL(c),
        LF_HETNET_BINDINGDB(c),
        LF_HETNET_PDSP_KI(c),
        LF_HETNET_US_PATENT(c),
        LF_HETNET_PUBCHEM(c)
    ]) else -1


# obtained from ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/ (ncbi's ftp server)
# https://github.com/dhimmel/entrez-gene/blob/a7362748a34211e5df6f2d185bb3246279760546/download/Homo_sapiens.gene_info.gz <-- use pandas and trim i guess
columns = [
    "tax_id", "GeneID", "Symbol",
    "LocusTag", "Synonyms", "dbXrefs",
    "chromosome", "map_location", "description",
    "type_of_gene", "Symbol_from_nomenclature_authority", "Full_name_from_nomenclature_authority",
    "Nomenclature_status", "Other_designations", "Modification_date"
]
gene_desc = pd.read_table("https://github.com/dhimmel/entrez-gene/blob/a7362748a34211e5df6f2d185bb3246279760546/download/Homo_sapiens.gene_info.gz?raw=true", sep="\t", names=columns, compression="gzip", skiprows=1)


def LF_CG_CHECK_GENE_TAG(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    sen = c[1].get_parent()
    gene_name = re.sub("\)", "", c[1].get_span().lower())
    gene_id = sen.entity_cids[c[1].get_word_start()]
    gene_entry_df = gene_desc.query("GeneID == @gene_id")

    if gene_entry_df.empty:
        return -1

    for token in gene_name.split(" "):
        if gene_entry_df["Symbol"].values[0].lower() == token or token in gene_entry_df["Synonyms"].values[0].lower():
            return 0
        elif token in gene_entry_df["description"].values[0].lower():
            return 0
    return -1

"""
SENTENCE PATTERN MATCHING
"""

binding_indication ={
    "binding", "binding site of", "heterodimerizes with",
    "reaction of", "binding of", "effects on", "by an inhibitor",
    "stimulated by", "reaction of", "can activate", "infusion of",
    "inhibited by", "receptor binding", "inhibitor(s)? of", "kinase inhibitors"
    "interaction of", "phosphorylation", "interacts with", "agonistic",
    "oxidation of", "oxidized to", "attack of"
}

weak_binding_indications = {
    "affected the activity of", "catalytic activity", "intermolecular interactions",
    "possible target protein", "local membraine effects", "response(s)? to"
}


upregulates = {
    "enhanced", "increased expression", "reversed the decreased.*(response)?",
    "maximial activation of", "increased expression of", "augmented",
    r"\bhigh\b", "elevate(d|s)?", "(significant(ly)?)? increase(d|s)?", "greated for",
    "greater in", "higher", "prevent their degeneration", "activate", "evoked a sustained rise"
}

downregulates = {
    "regulate transcription of", "inhibitors of", "kinase inhibitors",
    "negatively regulated by", "inverse agonist of", "downregulated", 
    "suppressed", "\blow\b", "reduce(d|s)?", "(significant(ly)?)? decrease(d|s)?",
    "inhibited by", "not higher", "unresponsive", "reduce", "antagonist", "inhibit(or|its)",
    "significantly attenuated"
}

gene_receivers = {
    "receptor", "(protein )?kinase",
    "antagonist", "agonist", "subunit", "binding", "bound"
}

compound_indentifiers = {
    "small molecules", "inhibitor"
}


def LF_CG_BINDING(c):
    """
    This label function is designed to look for phrases
    that imply a compound binding to a gene/protein
    """
    if re.search(ltp(binding_indication), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(binding_indication), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
        return 1
    elif re.search(ltp(binding_indication), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
        return 1
    else:
        return 0

def LF_CG_WEAK_BINDING(c):
    """
    This label function is designed to look for phrases
    that could imply a compound binding to a gene/protein
    """
    if re.search(ltp(weak_binding_indications), get_text_between(c), flags=re.I):
        return 1
    else:
        return 0

def LF_CG_UPREGULATES(c):
    """
    This label function is designed to look for phrases
    that implies a compound increaseing activity of a gene/protein
    """
    if re.search(ltp(upregulates), get_text_between(c), flags=re.I):
        return 1
    elif upregulates.intersection(get_left_tokens(c[1], window=2)):
        return 1
    else:
        return 0

def LF_CG_DOWNREGULATES(c):
    """
    This label function is designed to look for phrases
    that could implies a compound decreasing the activity of a gene/protein
    """
    if re.search(ltp(downregulates), get_text_between(c), flags=re.I):
        return 1
    elif downregulates.intersection(get_right_tokens(c[1], window=2)):
        return 1
    else:
        return 0

def LF_CG_GENE_RECEIVERS(c):
    """
    This label function is designed to look for phrases
    that imples a kinases or sort of protein that receives
    a stimulus to function
    """
    if re.search(ltp(gene_receivers), " ".join(get_right_tokens(c[1], window=4))) or re.search(ltp(gene_receivers), " ".join(get_left_tokens(c[1], window=4))):
        return 1
    elif re.search(ltp(gene_receivers), c[1].get_span(), flags=re.I):
        return 1
    else:
        return 0

def LF_CG_ASE_SUFFIX(c):
    """
    This label function is designed to look parts of the gene tags
    that implies a sort of "ase" or enzyme
    """
    if re.search(r"ase\b", c[1].get_span(), flags=re.I):
        return 1
    else:
        return 0

def LF_CG_IN_SERIES(c):
    """
    This label function is designed to look for a mention being caught
    in a series of other genes or compounds
    """
    if len(re.findall(r',', get_tagged_text(c))) >= 2:
        if re.search(', and', get_tagged_text(c)):
            return -1
    return 0

def LF_CG_ANTIBODY(c):
    """
    This label function is designed to look for phrase
    antibody.
    """
    if "antibody" in c[1].get_span() or re.search("antibody", " ".join(get_right_tokens(c[1], window=3))):
        return 1
    elif "antibodies" in c[1].get_span() or re.search("antibodies", " ".join(get_right_tokens(c[1], window=3))):
        return 1
    else:
        return 0


method_indication = {
    "investigated (the effect of|in)", "was assessed by", "assessed", 
    "compared with", "compared to", "were analyzed", "evaluated in", "examination of", "examined in",
    "quantified in" "quantification by", "we review", "was measured", "we(re)? studied", 
    "we measured", "derived from", "Regulation of", "(are|is) discussed", "to measure", "to study",
    "to explore", "detection of", "authors summarize", "responsiveness of",
    "used alone", "blunting of", "measurement of", "detection of", "occurence of", 
    "our objective was", "to test the hypothesis", "studied in", "were reviewed",
    "randomized study", "this report considers", "was administered", "determinations of",
    "we examine", "we evaluated", "to establish", "were selected", "authors determmined",
    "we investigated", "to assess", "analyses were done", "useful tool for the study of", r"^The effect of",
    }


def LF_CG_METHOD_DESC(c):
    """
    This label function is designed to look for phrases 
    that imply a sentence is description an experimental design
    """
    if re.search(ltp(method_indication), get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_CG_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum([LF_CG_BINDING(c), LF_CG_WEAK_BINDING(c), 
        LF_CG_GENE_RECEIVERS(c), LF_CG_ANTIBODY(c), 
        LF_CG_UPREGULATES(c),  LF_CG_DOWNREGULATES(c)])
    negative_num = np.abs(np.sum(LF_CG_METHOD_DESC(c)))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_CG_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_CG_NO_CONCLUSION(c):
        return 1
    else:
        return 0

def LF_CG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't right next to each other.
    """
    return -1 if len(list(get_between_tokens(c))) <= 2 else 0

def LF_CG_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't too far from each other.
    """
    return -1 if len(list(get_between_tokens(c))) > 25 else 0

def LF_CG_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention are in an acceptable distance between 
    each other
    """
    return 0 if any([
        LF_CG_DISTANCE_LONG(c),
        LF_CG_DISTANCE_SHORT(c)
        ]) else 1 if random.random() < 0.65 else 0

def LF_CG_NO_VERB(c):
    """
    This label function is designed to fire if a given
    sentence doesn't contain a verb. Helps cut out some of the titles
    hidden in Pubtator abstracts
    """
    if len([x for x in  nltk.pos_tag(word_tokenize(c.get_parent().text)) if "VB" in x[1]]) == 0:
        if "correlates with" in c.get_parent().text:
            return 0
        return -1
    return 0

def LF_CG_PARENTHETICAL_DESC(c):
    """
    This label function looks for mentions that are in paranthesis.
    Some of the gene mentions are abbreviations rather than names of a gene.
    """
    if ")" in c[1].get_span() and "(" in list(get_left_tokens(c[1], window=1)):
        if LF_CG_DISTANCE_SHORT(c):
            return -1
    return 0


"""
Bi-Clustering LFs
"""
path = pathlib.Path(__file__).joinpath("../../../../../dependency_cluster/chemical_gene_bicluster_results.tsv.xz").resolve()
bicluster_dep_df = pd.read_table(path)

binds_base = set([tuple(x) for x in bicluster_dep_df.query("B>0")[["pubmed_id", "sentence_num"]].values])
agonism_base = set([tuple(x) for x in bicluster_dep_df[bicluster_dep_df["A+"] > 0][["pubmed_id", "sentence_num"]].values])
antagonism_base = set([tuple(x) for x in bicluster_dep_df[bicluster_dep_df["A-"] > 0][["pubmed_id", "sentence_num"]].values])
inc_expression_base = set([tuple(x) for x in bicluster_dep_df[bicluster_dep_df["E+"] > 0][["pubmed_id", "sentence_num"]].values])
dec_expression_base = set([tuple(x) for x in bicluster_dep_df[bicluster_dep_df["E-"] > 0][["pubmed_id", "sentence_num"]].values])
aff_expression_base = set([tuple(x) for x in bicluster_dep_df.query("E>0")[["pubmed_id", "sentence_num"]].values])
inhibits_base = set([tuple(x) for x in bicluster_dep_df.query("N>0")[["pubmed_id", "sentence_num"]].values])

def LF_CG_BICLUSTER_BINDS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in binds_base:
        return 1
    return 0

def LF_CG_BICLUSTER_AGONISM(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in agonism_base:
        return 1
    return 0

def LF_CG_BICLUSTER_ANTAGONISM(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in antagonism_base:
        return 1
    return 0

def LF_CG_BICLUSTER_INC_EXPRESSION(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in inc_expression_base:
        return 1
    return 0

def LF_CG_BICLUSTER_DEC_EXPRESSION(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in dec_expression_base:
        return 1
    return 0

def LF_CG_BICLUSTER_AFF_EXPRESSION(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in aff_expression_base:
        return 1
    return 0

def LF_CG_BICLUSTER_INHIBITS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in inhibits_base:
        return 1
    return 0

"""
RETRUN LFs to Notebook
"""

CG_LFS = {
    "CbG":
    OrderedDict({
        "LF_HETNET_DRUGBANK": LF_HETNET_DRUGBANK,
        "LF_HETNET_DRUGCENTRAL": LF_HETNET_DRUGCENTRAL,
        "LF_HETNET_ChEMBL": LF_HETNET_ChEMBL,
        "LF_HETNET_BINDINGDB": LF_HETNET_BINDINGDB,
        "LF_HETNET_PDSP_KI": LF_HETNET_PDSP_KI,
        "LF_HETNET_US_PATENT": LF_HETNET_US_PATENT,
        "LF_HETNET_PUBCHEM": LF_HETNET_PUBCHEM,
        "LF_HETNET_CG_ABSENT":LF_HETNET_CG_ABSENT,
        "LF_CG_CHECK_GENE_TAG": LF_CG_CHECK_GENE_TAG,
        "LF_CG_BINDING": LF_CG_BINDING,
        "LF_CG_WEAK_BINDING": LF_CG_WEAK_BINDING,
        "LF_CG_GENE_RECEIVERS": LF_CG_GENE_RECEIVERS,
        "LF_CG_ASE_SUFFIX": LF_CG_ASE_SUFFIX,
        "LF_CG_IN_SERIES": LF_CG_IN_SERIES,
        "LF_CG_ANTIBODY": LF_CG_ANTIBODY,
        "LF_CG_METHOD_DESC": LF_CG_METHOD_DESC,
        "LF_CG_NO_CONCLUSION": LF_CG_NO_CONCLUSION,
        "LF_CG_CONCLUSION": LF_CG_CONCLUSION,
        "LF_CG_DISTANCE_SHORT": LF_CG_DISTANCE_SHORT,
        "LF_CG_DISTANCE_LONG": LF_CG_DISTANCE_LONG,
        "LF_CG_ALLOWED_DISTANCE": LF_CG_ALLOWED_DISTANCE,
        "LF_CG_NO_VERB": LF_CG_NO_VERB,
        "LF_CG_BICLUSTER_BINDS":LF_CG_BICLUSTER_BINDS,
        "LF_CG_BICLUSTER_AGONISM":LF_CG_BICLUSTER_AGONISM,
        "LF_CG_BICLUSTER_ANTAGONISM":LF_CG_BICLUSTER_ANTAGONISM,
        "LF_CG_BICLUSTER_INC_EXPRESSION":LF_CG_BICLUSTER_INC_EXPRESSION,
        "LF_CG_BICLUSTER_DEC_EXPRESSION":LF_CG_BICLUSTER_DEC_EXPRESSION,
        "LF_CG_BICLUSTER_AFF_EXPRESSION":LF_CG_BICLUSTER_AFF_EXPRESSION,
        "LF_CG_BICLUSTER_INHIBITS":LF_CG_BICLUSTER_INHIBITS,
    }),
    "CuG":OrderedDict({
        "LF_CG_UPREGULATES": LF_CG_UPREGULATES,
    }),
    "CdG":OrderedDict({
        "LF_CG_DOWNREGULATES": LF_CG_DOWNREGULATES,
    })
}
