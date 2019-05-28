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
path = pathlib.Path(__file__).joinpath('../../../../compound_treats_disease.tsv.xz').resolve()
pair_df = pd.read_table(path, dtype={"sources": str})
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        source = re.sub(r' \(\w+\)', '', source)
        key = row.drugbank_id, row.doid_id, source
        knowledge_base.add(key)

def LF_HETNET_PHARMACOTHERAPYDB(c):
    return 1 if (c.Compound_cid, c.Disease_cid, "pharmacotherapydb") in knowledge_base else 0

def LF_HETNET_CD_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear 
    in the databases above.
    """
    return 0 if any([
        LF_HETNET_PHARMACOTHERAPYDB(c)
    ]) else -1


disease_normalization_df = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/slim-terms-prop.tsv")
wordnet_lemmatizer = WordNetLemmatizer()

def LF_CD_CHECK_DISEASE_TAG(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    sen = c[1].get_parent()
    disease_name = re.sub("\) ?", "", c[1].get_span())
    disease_name = re.sub(r"(\w)-(\w)", r"\g<1> \g<2>", disease_name)
    disease_name = " ".join([word for word in word_tokenize(disease_name) if word not in set(stop_word_list)])

    # If abbreviation skip since no means of easy resolution
    if len(disease_name) <=5 and disease_name.isupper():
        return 0

    disease_id = sen.entity_cids[c[1].get_word_start()]
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

treat_indication = {
    "was administered", "treated with", "treatment of", 
    "treatment for", "feeding a comparable dose.*of",
    "was given", "were injected", "be administered", "treatment with",
    "to treat", "induced regression of", "therapy", "treatment in active",
    "chemotherapeutic( effect of )?", "protection from",
    "promising drug for treatment-refractory", "inhibit the proliferation of",
    "monotherapy", "treatment of"
}

incorrect_depression_indication = {
    "sharp", "produced a", "opiate", "excitability", 
    "produced by", "causes a", "was", "effects of",
    "long lasting {{B}} of", "led to a"
}

weak_treatment_indications = {
    "responds to", "effective with", "inhibition of",
    "hypotensive efficacy of", "was effecitve in", "treatment", 
    "render the disease sero-negative", "enhance the activity of",
    "blocked in", "(possible|potential) benefits of", "antitumor activity",
    "prolactin response"
}

incorrect_compound_indications = {
    "human homolog", "levels", "pathway", "receptor(-subtype)?"
}

palliates_indication = {
    "prophylatic effect", "supression of", "significant reduction in",
    "therapy in", "inhibited", "prevents or retards", "pulmonary vasodilators",
    "management of", "was controlled with"
}

compound_indications = {
    "depleting agents", "blocking agents", "antagonist",
    "antirheumatric drugs", "agonist"
}

trial_indications = {
    "clinical trial", "(controlled, )?single-blind trial",
    "double-blind.+trial", "multi-centre trial",
    "double-blind", "placebo", "trial(s)?", "randomized"
}


def LF_CtD_TREATS(c):
    if re.search(ltp(treat_indication), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(treat_indication), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
        return 1
    elif re.search(ltp(treat_indication), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
        return 1
    else:
        return 0

def LF_CD_CHECK_DEPRESSION_USAGE(c):
    if "depress" in c[1].get_span():
        if re.search(ltp(incorrect_depression_indication), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
            return -1
        elif re.search(ltp(incorrect_depression_indication), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
            return -1    
    return 0

def LF_CtD_WEAKLY_TREATS(c):
    """
    This label function is designed to look for phrases
    that have a weak implication towards a compound treating a disease
    """
    if re.search(ltp(weak_treatment_indications), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(weak_treatment_indications), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
        return 1
    elif re.search(ltp(weak_treatment_indications), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
        return 1
    else:
        return 0

def LF_CD_INCORRECT_COMPOUND(c):
    """
    This label function is designed to capture phrases
    that indicate the mentioned compound is a protein not a drug
    """
    if re.search(ltp(incorrect_compound_indications), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
        return -1
    elif re.search(ltp(incorrect_compound_indications), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
        return -1
    else:
        return 0

def LF_CpD_PALLIATES(c):
    """
    This label function is designed to look for phrases
    that could imply a compound binding to a gene/protein
    """
    if re.search(ltp(palliates_indication), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(palliates_indication), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
        return 1
    elif re.search(ltp(palliates_indication), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
        return 1
    else:
        return 0

def LF_CtD_COMPOUND_INDICATION(c):
    """
    This label function is designed to look for phrases
    that implies a compound increaseing activity of a gene/protein
    """
    if re.search(ltp(compound_indications), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(compound_indications), " ".join(get_left_tokens(c[0], window=5)), flags=re.I):
        return 1
    elif re.search(ltp(compound_indications), " ".join(get_right_tokens(c[0], window=5)), flags=re.I):
        return 1
    else:
        return 0

def LF_CtD_TRIAL(c):
    return 1 if re.search(ltp(trial_indications), get_tagged_text(c), flags=re.I) else 0

def LF_CD_IN_SERIES(c):
    """
    This label function is designed to look for a mention being caught
    in a series of other genes or compounds
    """
    if len(re.findall(r',', get_tagged_text(c))) >= 2:
        if re.search(', and', get_tagged_text(c)):
            return -1
    if re.search(r"\(a\)|\(b\)|\(c\)", get_tagged_text(c)):
        return -1
    return 0


method_indication = {
    "investigated (the effect of|in)", "was assessed by", "assessed", 
    "compared with", "compared to", "were analyzed", "evaluated in", "examination of", "examined in",
    "quantified in" "quantification by", "we review", "was measured", "we(re)? studied", 
    "we measured", "derived from", "regulation of", "(are|is) discussed", "to measure", "to study",
    "to explore", "detection of", "authors summarize", "responsiveness of",
    "used alone", "blunting of", "measurement of", "detection of", "occurence of", 
    "our objective was", "to test the hypothesis", "studied in", "were reviewed",
    "randomized study", "this report considers", "was administered", "determinations of",
    "we examine", "we evaluated", "to establish", "were selected", "authors determmined",
    "we investigated", "to assess", "analyses were done", "useful tool for the study of", r"^The effect of",
    "were investigated", "to evaluate", "study was conducted", "to assess", "authors applied",
    "were determined"
    }

title_indication = {
    "impact of", "effects of", "the use of", "understanding {{B}}( preconditioning)?"

}

def LF_CD_METHOD_DESC(c):
    """
    This label function is designed to look for phrases 
    that imply a sentence is description an experimental design
    """
    if re.search(ltp(method_indication), get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_CD_TITLE(c):
    """
    This label function is designed to look for phrases 
    that imply a sentence is the title
    """
    if re.search(r'^(\[|\[ )?'+ltp(title_indication), get_tagged_text(c), flags=re.I):
        return -1
    elif re.search(ltp(title_indication)+r'$', get_tagged_text(c), flags=re.I):
        return -1
    elif "(author's transl)" in get_tagged_text(c):
        return -1
    elif ":" in get_between_tokens(c):
        return -1
    else:
        return 0

def LF_CtD_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum([
        LF_CtD_TREATS(c),
        LF_CD_CHECK_DEPRESSION_USAGE(c),
        LF_CtD_WEAKLY_TREATS(c),
        LF_CtD_COMPOUND_INDICATION(c),
        LF_CtD_TRIAL(c)])
    negative_num = np.abs(np.sum([LF_CD_METHOD_DESC(c), LF_CD_IN_SERIES(c)]))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_CtD_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_CtD_NO_CONCLUSION(c):
        return 1
    else:
        return 0

def LF_CD_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't right next to each other.
    """
    return -1 if len(list(get_between_tokens(c))) <= 2 else 0

def LF_CD_DISTANCE_LONG(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention aren't too far from each other.
    """
    return -1 if len(list(get_between_tokens(c))) > 25 else 0

def LF_CD_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the compound mention
    and the gene mention are in an acceptable distance between 
    each other
    """
    return 0 if any([
        LF_CD_DISTANCE_LONG(c),
        LF_CD_DISTANCE_SHORT(c)
        ]) else 1

def LF_CD_NO_VERB(c):
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


"""
Bi-Clustering LFs
"""
path = pathlib.Path(__file__).joinpath("../../../../../biclustering/compound_disease_bicluster_results.tsv.xz").resolve()
bicluster_dep_df = pd.read_table(path)
treatment_base = set([tuple(x) for x in bicluster_dep_df.query("T>0")[["pubmed_id", "sentence_num"]].values])
inhibits_base = set([tuple(x) for x in bicluster_dep_df.query("C>0")[["pubmed_id", "sentence_num"]].values])
side_effect_base = set([tuple(x) for x in bicluster_dep_df.query("Sa>0")[["pubmed_id", "sentence_num"]].values])
prevents_base = set([tuple(x) for x in bicluster_dep_df.query("Pr>0")[["pubmed_id", "sentence_num"]].values])
alleviates_base = set([tuple(x) for x in bicluster_dep_df.query("Pa>0")[["pubmed_id", "sentence_num"]].values])
disease_role_base = set([tuple(x) for x in bicluster_dep_df.query("J>0")[["pubmed_id", "sentence_num"]].values])
biomarkers_base = set([tuple(x) for x in bicluster_dep_df.query("Mp>0")[["pubmed_id", "sentence_num"]].values])

def LF_CD_BICLUSTER_TREATMENT(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in treatment_base:
        return 1
    return 0

def LF_CD_BICLUSTER_INHIBITS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in inhibits_base:
        return 1
    return 0

def LF_CD_BICLUSTER_SIDE_EFFECT(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in side_effect_base:
        return 1
    return 0

def LF_CD_BICLUSTER_PREVENTS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in prevents_base:
        return 1
    return 0

def LF_CD_BICLUSTER_ALLEVIATES(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in alleviates_base:
        return 1
    return 0

def LF_CD_BICLUSTER_DISEASE_ROLE(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in disease_role_base:
        return 1
    return 0

def LF_CD_BICLUSTER_BIOMARKERS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in biomarkers_base:
        return 1
    return 0

"""
RETRUN LFs to Notebook
"""

CD_LFS = {
    "CtD":OrderedDict({
        "LF_HETNET_PHARMACOTHERAPYDB":LF_HETNET_PHARMACOTHERAPYDB,
        "LF_HETNET_CD_ABSENT":LF_HETNET_CD_ABSENT,
        "LF_CD_CHECK_DISEASE_TAG": LF_CD_CHECK_DISEASE_TAG,
        "LF_CtD_TREATS": LF_CtD_TREATS,
        "LF_CD_CHECK_DEPRESSION_USAGE": LF_CD_CHECK_DEPRESSION_USAGE,
        "LF_CtD_WEAKLY_TREATS": LF_CtD_WEAKLY_TREATS,
        "LF_CD_INCORRECT_COMPOUND":LF_CD_INCORRECT_COMPOUND,
        "LF_CtD_COMPOUND_INDICATION": LF_CtD_COMPOUND_INDICATION,
        "LF_CtD_TRIAL": LF_CtD_TRIAL,
        "LF_CD_IN_SERIES": LF_CD_IN_SERIES,
        "LF_CD_METHOD_DESC": LF_CD_METHOD_DESC,
        "LF_CD_TITLE": LF_CD_TITLE,
        "LF_CtD_NO_CONCLUSION": LF_CtD_NO_CONCLUSION,
        "LF_CtD_CONCLUSION": LF_CtD_CONCLUSION,
        "LF_CD_DISTANCE_SHORT": LF_CD_DISTANCE_SHORT,
        "LF_CD_DISTANCE_LONG": LF_CD_DISTANCE_LONG,
        "LF_CD_ALLOWED_DISTANCE": LF_CD_ALLOWED_DISTANCE,
        "LF_CD_NO_VERB": LF_CD_NO_VERB,
        "LF_CD_BICLUSTER_TREATMENT":LF_CD_BICLUSTER_TREATMENT,
        "LF_CD_BICLUSTER_INHIBITS":LF_CD_BICLUSTER_INHIBITS,
        "LF_CD_BICLUSTER_SIDE_EFFECT":lambda x: -1*LF_CD_BICLUSTER_SIDE_EFFECT(x),
        "LF_CD_BICLUSTER_PREVENTS":LF_CD_BICLUSTER_PREVENTS,
        "LF_CD_BICLUSTER_ALLEVIATES": lambda x: -1*LF_CD_BICLUSTER_ALLEVIATES(x),
        "LF_CD_BICLUSTER_DISEASE_ROLE":LF_CD_BICLUSTER_DISEASE_ROLE,
        "LF_CD_BICLUSTER_BIOMARKERS":LF_CD_BICLUSTER_BIOMARKERS
    }),
    "CpD":OrderedDict({
        "LF_CpD_PALLIATES": LF_CpD_PALLIATES,
        "LF_CD_INCORRECT_COMPOUND":LF_CD_INCORRECT_COMPOUND,
        "LF_CD_BICLUSTER_ALLEVIATES":LF_CD_BICLUSTER_ALLEVIATES,
        "LF_CD_BICLUSTER_INHIBITS":LF_CD_BICLUSTER_INHIBITS,
        "LF_CD_BICLUSTER_PREVENTS":LF_CD_BICLUSTER_PREVENTS,
        "LF_CD_BICLUSTER_DISEASE_ROLE":LF_CD_BICLUSTER_DISEASE_ROLE,
        "LF_CD_BICLUSTER_BIOMARKERS":LF_CD_BICLUSTER_BIOMARKERS
    })
}
