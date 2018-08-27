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
path = pathlib.Path(__file__).joinpath('../../data/disease-gene-pairs-association.csv.xz').resolve()
pair_df = pd.read_csv(path, dtype={"sources": str})
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        key = str(row.entrez_gene_id), row.doid_id, source
        knowledge_base.add(key)

def LF_HETNET_DISEASES(c):
    return 1 if (c.Gene_cid, c.Disease_cid, "DISEASES") in knowledge_base else 0

def LF_HETNET_DOAF(c):
    return 1 if (c.Gene_cid, c.Disease_cid, "DOAF") in knowledge_base else 0

def LF_HETNET_DisGeNET(c):
    return 1 if (c.Gene_cid, c.Disease_cid, "DisGeNET") in knowledge_base else 0

def LF_HETNET_GWAS(c):
    return 1 if (c.Gene_cid, c.Disease_cid, "GWAS Catalog") in knowledge_base else 0

def LF_HETNET_DG_ABSENT(c):
    return 0 if any([
        LF_HETNET_DISEASES(c),
        LF_HETNET_DOAF(c),
        LF_HETNET_DisGeNET(c),
        LF_HETNET_GWAS(c)
    ]) else -1


# obtained from ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/ (ncbi's ftp server)
# https://github.com/dhimmel/entrez-gene/blob/a7362748a34211e5df6f2d185bb3246279760546/download/Homo_sapiens.gene_info.gz <-- use pandas and trim i guess
gene_desc = pd.read_table("gene_desc.tsv")


def LF_DG_CHECK_GENE_TAG(c):
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


#disease_desc = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv")
disease_normalization_df = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/slim-terms-prop.tsv")
wordnet_lemmatizer = WordNetLemmatizer()

def LF_DG_CHECK_DISEASE_TAG(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    sen = c[0].get_parent()
    disease_name = re.sub("\) ?", "", c[0].get_span())
    disease_name = re.sub(r"(\w)-(\w)", r"\g<1> \g<2>", disease_name)
    disease_name = " ".join([word for word in word_tokenize(disease_name) if word not in set(stopwords.words('english'))])

    # If abbreviation skip since no means of easy resolution
    if len(disease_name) <=5 and disease_name.isupper():
        return 0

    disease_id = sen.entity_cids[c[0].get_word_start()]
    result = disease_normalization_df[
        disease_normalization_df["subsumed_name"].str.contains(disease_name.lower(), regex=False)
    ]
    # If no match then return -1
    if result.empty:

        # check the reverse direction e.g. carcinoma lung -> lung carcinoma
        disease_name_tokens = word_tokenize(disease_name)
        if len(disease_name_tokens) == 2:
            result = disease_normalization_df[
                disease_normalization_df["subsumed_name"].str.contains(" ".join(disease_name_tokens[-1::0-1]).lower(), regex=False)
            ]
            
            # if reversing doesn't work then output -t
            if not result.empty:
                slim_id = result['slim_id'].values[0]
                if slim_id == disease_id:
                    return 0
        return -1
    else:
        # If it can be normalized return 0 else -1
        slim_id = result['slim_id'].values[0]
        if slim_id == disease_id:
            return 0
        else:
            return -1

"""
SENTENCE PATTERN MATCHING
"""

biomarker_indicators = {
    "useful marker of", "useful in predicting","modulates the expression of", "expressed in",
    "prognostic marker", "tissue marker", "tumor marker", "level(s)? (of|in)", 
    "high concentrations of", "(cytoplamsic )?concentration of",
    "have fewer", "quantification of", "evaluation of", "hypersecreted by",
    "assess the presensece of", "stained postively for", "overproduced", 
    "prognostic factor", "characterized by a marked",
    "plasma levels of", "had elevated", "were detected", "exaggerated response to", 
    "serum", "expressed on", "overexpression of"
    }

direct_association = {
    "association with", "association between", "associated with", "associated between", 
    "stimulated by", "correlat(ed|es|ion)? between", "correlat(ed|es|ion)? with",
    "significant ultradian variation", "showed (that|loss)", "found in", "involved in","central role in",
    "inhibited by", "greater for", "indicative of","increased production of",
    "control the extent of", "secreted by", "detected in", "positive for", "to be mediated", 
    "was produced by", "stimulates", "precipitated by", "affects", "counteract cholinergic deficits", 
    "mediator of", "candidate gene", "categorized", "positive correlation", 
    "regulated by", "important role in", "significant amounts of", "to contain"
    }

positive_direction = {
    r"\bhigh\b", "elevate(d|s)?", "(significant(ly)?)? increase(d|s)?", "greated for",
    "greater in", "higher", "prevent their degeneration"
}

negative_direction = {
    r"\blow\b", "reduce(d|s)?", "(significant(ly)?)? decrease(d|s)?", "inhibited by", "not higher",
    "unresponsive"
}

diagnosis_indicators = {
    "prognostic significance of", "prognostic indicator for", "prognostic cyosolic factor",
    "prognostic parameter for", "prognostic information for", "predict(or|ive) of",
    "predictor of prognosis in", "indicative of", "diagnosis of", "was positive for",
    "detection of", "determined by", "diagnositic sensitivity", "dianostic specificity",
    "prognostic factor", "variable for the identification", "potential therapeutic agent",
    "prognostic parameter for", "identification of"
}

no_direct_association = {
    "not significant", "not significantly", "no association", "not associated",
    "no correlation between" "no correlation in", "no correlation with", "not correlated with",
     "not detected in", "not been observed", "not appear to be related to", "neither", 
     "provide evidence against", "not a constant", "not predictive", "nor were they correlated with"
    }

weak_association = {
    "not necessarily indicate", "the possibility", "low correlation", "may be.* important", "might facillitate"
}

negative_indication =  {
    "failed", "poorly"
}

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

title_indication = {
    "Effect of", "Evaluation of", 
    "Clincal value of", "Extraction of",
    "Responsiveness of", "The potential for",
    "as defined by immunohistochemistry", "Comparison between",
    "Characterization of", "A case of", "Occurrence of",
    "Inborn", "Episodic", "Detection of", "Immunostaining of"
}

def LF_DG_IS_BIOMARKER(c):
    """
    This label function examines a sentences to determine of a sentence
    is talking about a biomarker. (A biomarker leads towards D-G assocation
    c - The candidate obejct being passed in
    """
    if re.search(ltp(biomarker_indicators) + r".*{{B}}", get_tagged_text(c), flags=re.I):
        return 1
    elif re.search(r"{{B}}.*" + ltp(biomarker_indicators), get_tagged_text(c), flags=re.I):
        return 1
    else:
        return 0

def LF_DG_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is an association.
    """
    if re.search(r'(?<!not )(?<!no )' + ltp(direct_association), get_text_between(c), flags=re.I):
        return 1
    elif re.search(r'(?<!not )(?<!no )' + ltp(direct_association) + r".*({{B}}|{{A}})", get_tagged_text(c), flags=re.I):
        return 1
    elif re.search(r"({{B}}|{{A}}).*(?<!not )(?<!no )" + ltp(direct_association), get_tagged_text(c), flags=re.I):
        return 1
    else:
        return 0

def LF_DG_WEAK_ASSOCIATION(c):
    if re.search(ltp(weak_association), get_text_between(c), flags=re.I):
        return -1
    elif re.search(ltp(weak_association) + r".*({{B}}|{{A}})", get_tagged_text(c), flags=re.I):
        return -1
    elif re.search(r"({{B}}|{{A}}).*" + ltp(weak_association), get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_DG_NO_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is no an association.
    """
    if re.search(ltp(no_direct_association), get_text_between(c), flags=re.I):
        return -1
    elif re.search(ltp(no_direct_association) + r".*({{B}}|{{A}})", get_tagged_text(c), flags=re.I):
        return -1
    elif re.search(r"({{B}}|{{A}}).*" + ltp(no_direct_association), get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_DG_METHOD_DESC(c):
    if re.search(ltp(method_indication), get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_DG_TITLE(c):
    if re.search(r'^'+ltp(title_indication), get_tagged_text(c), flags=re.I):
        return -1
    elif re.search(ltp(title_indication)+r'$', get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_DG_POSITIVE_DIRECTION(c):
    return 1 if any([rule_regex_search_btw_AB(c, r'.*'+ltp(positive_direction)+r'.*', 1), rule_regex_search_btw_BA(c, r'.*'+ltp(positive_direction)+r'.*', 1)]) or \
        re.search(r'({{A}}|{{B}}).*({{A}}|{{B}}).*' + ltp(positive_direction), get_tagged_text(c)) else 0

def LF_DG_NEGATIVE_DIRECTION(c):
    return 1 if any([rule_regex_search_btw_AB(c, r'.*'+ltp(negative_direction)+r'.*', 1), rule_regex_search_btw_BA(c, r'.*'+ltp(negative_direction)+r'.*', 1)]) or  \
        re.search(r'({{A}}|{{B}}).*({{A}}|{{B}}).*' + ltp(positive_direction), get_tagged_text(c)) else 0

def LF_DG_DIAGNOSIS(c):
    return 1 if any([rule_regex_search_btw_AB(c, r'.*'+ltp(diagnosis_indicators) + r".*", 1), rule_regex_search_btw_BA(c, r'.*'+ltp(diagnosis_indicators) + r".*", 1)]) or  \
        re.search(r'({{A}}|{{B}}).*({{A}}|{{B}}).*' + ltp(diagnosis_indicators), get_tagged_text(c)) else 0

def LF_DG_RISK(c):
    return 1 if re.search(r"risk (of|for)", get_tagged_text(c), flags=re.I) else 0

def LF_DG_PATIENT_WITH(c):
    return 1 if re.search(r"patient(s)? with {{A}}", get_tagged_text(c), flags=re.I) else 0

def LF_DG_PURPOSE(c):
    return -1 if "PURPOSE:" in get_tagged_text(c) else 0

def LF_DG_CONCLUSION_TITLE(c):
    return 1 if "CONCLUSION" in get_tagged_text(c) or "concluded" in get_tagged_text(c) else 0

def LF_DG_NO_CONCLUSION(c):
    positive_num = np.sum([LF_DG_ASSOCIATION(c), LF_DG_IS_BIOMARKER(c),LF_DG_NO_ASSOCIATION(c),  
            LF_DG_POSITIVE_DIRECTION(c), LF_DG_NEGATIVE_DIRECTION(c), LF_DG_DIAGNOSIS(c),
            np.abs(LF_DG_WEAK_ASSOCIATION(c)), np.abs(LF_DG_NO_ASSOCIATION(c))])
    negative_num = np.abs(np.sum(LF_DG_METHOD_DESC(c), LF_DG_TITLE(c)))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_DG_CONCLUSION(c):
    if LF_DG_NO_ASSOCIATION(c) or LF_DG_WEAK_ASSOCIATION(c):
        return -1
    elif not LF_DG_NO_CONCLUSION(c):
        return 1
    else:
        return 0

def LF_DG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't right next to each other.
    """
    return -1 if len(list(get_between_tokens(c))) <= 2 else 0

def LF_DG_DISTANCE_LONG(c):
    return -1 if len(list(get_between_tokens(c))) > 25 else 0

def LF_DG_ALLOWED_DISTANCE(c):
    return 0 if any([
        LF_DG_DISTANCE_LONG(c),
        LF_DG_DISTANCE_SHORT(c)
        ]) else 1 if random.random() < 0.65 else 0

def LF_DG_NO_VERB(c):
    # Work on adding preprocessing steps
    if len([x for x in  nltk.pos_tag(word_tokenize(c.get_parent().text)) if "VB" in x[1]]) == 0:
        if "correlates with" in c.get_parent().text:
            return 0
        return -1
    return 0

"""
Bi-Clustering LFs
"""
bicluster_dep_df = pd.read_table("data/disease_gene/disease_associates_gene/biclustering/disease_gene_bicluster_results.tsv")

def LF_DG_BICLUSTER_CASUAL_MUTATIONS(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["U"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_MUTATIONS(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["Ud"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_DRUG_TARGETS(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["D"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_PATHOGENESIS(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["J"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_THERAPEUTIC(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["Te"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_POLYMORPHISMS(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["Y"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_PROGRESSION(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["G"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_BIOMARKERS(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["Md"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_OVEREXPRESSION(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["X"].sum() > 0.0:
            return 1
    return 0

def LF_DG_BICLUSTER_REGULATION(c):
    sen_pos = c.get_parent().position
    pubmed_id = c.get_parent().document.name
    query = bicluster_dep_df.query("pubmed_id==@pubmed_id&sentence_num==@sen_pos")
    if not(query.empty):
        if query["L"].sum() > 0.0:
            return 1
    return 0

"""
RETRUN LFs to Notebook
"""

DG_LFS = {
    "DaG_DB": 
    {
        "LF_HETNET_DISEASES": LF_HETNET_DISEASES,
        "LF_HETNET_DOAF": LF_HETNET_DOAF,
        "LF_HETNET_DisGeNET": LF_HETNET_DisGeNET,
        "LF_HETNET_GWAS": LF_HETNET_GWAS,
        "LF_HETNET_DG_ABSENT":LF_HETNET_DG_ABSENT,
        "LF_DG_CHECK_GENE_TAG": LF_DG_CHECK_GENE_TAG, 
        "LF_DG_CHECK_DISEASE_TAG": LF_DG_CHECK_DISEASE_TAG
    },
    
    "DaG_TEXT":
    {
        "LF_DG_IS_BIOMARKER": LF_DG_IS_BIOMARKER,
        "LF_DG_ASSOCIATION": LF_DG_ASSOCIATION,
        "LF_DG_WEAK_ASSOCIATION": LF_DG_WEAK_ASSOCIATION,
        "LF_DG_NO_ASSOCIATION": LF_DG_NO_ASSOCIATION,
        "LF_DG_METHOD_DESC": LF_DG_METHOD_DESC,
        "LF_DG_TITLE": LF_DG_TITLE,
        "LF_DG_POSITIVE_DIRECTION": LF_DG_POSITIVE_DIRECTION,
        "LF_DG_NEGATIVE_DIRECTION": LF_DG_NEGATIVE_DIRECTION,
        "LF_DG_DIAGNOSIS": LF_DG_DIAGNOSIS,
        "LF_DG_RISK": LF_DG_RISK,
        "LF_DG_PATIENT_WITH":LF_DG_PATIENT_WITH,
        "LF_DG_PURPOSE":LF_DG_PURPOSE,
        "LF_DG_CONCLUSION_TITLE":LF_DG_CONCLUSION_TITLE,
        "LF_DG_NO_CONCLUSION": LF_DG_NO_CONCLUSION,
        "LF_DG_CONCLUSION": LF_DG_CONCLUSION,
        "LF_DG_DISTANCE_SHORT": LF_DG_DISTANCE_SHORT,
        "LF_DG_DISTANCE_LONG": LF_DG_DISTANCE_LONG,
        "LF_DG_ALLOWED_DISTANCE": LF_DG_ALLOWED_DISTANCE,
        "LF_DG_NO_VERB": LF_DG_NO_VERB
    },
    
    "DG_BICLUSTER":
    {
        "LF_DG_BICLUSTER_CASUAL_MUTATIONS":LF_DG_BICLUSTER_CASUAL_MUTATIONS,
        "LF_DG_BICLUSTER_MUTATIONS": LF_DG_BICLUSTER_MUTATIONS,
        "LF_DG_BICLUSTER_DRUG_TARGETS": LF_DG_BICLUSTER_DRUG_TARGETS,
        "LF_DG_BICLUSTER_PATHOGENESIS": LF_DG_BICLUSTER_PATHOGENESIS,
        "LF_DG_BICLUSTER_THERAPEUTIC": LF_DG_BICLUSTER_THERAPEUTIC,
        "LF_DG_BICLUSTER_POLYMORPHISMS": LF_DG_BICLUSTER_POLYMORPHISMS,
        "LF_DG_BICLUSTER_PROGRESSION": LF_DG_BICLUSTER_PROGRESSION,
        "LF_DG_BICLUSTER_BIOMARKERS": LF_DG_BICLUSTER_BIOMARKERS,
        "LF_DG_BICLUSTER_OVEREXPRESSION": LF_DG_BICLUSTER_OVEREXPRESSION,
        "LF_DG_BICLUSTER_REGULATION": LF_DG_BICLUSTER_REGULATION
    }     
}
