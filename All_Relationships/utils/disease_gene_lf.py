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
import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    prin()
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
hetnet_kb = pd.read_csv("hetnet_dg_kb.csv")


def LF_HETNETS(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    if not hetnet_kb[(hetnet_kb["disease_id"] == str(c.Disease_cid)) & (hetnet_kb["gene_id"] == int(c.Gene_cid))].empty:
        return 1
    else:
        return -1


# obtained from ftp://ftp.ncbi.nlm.nih.gov/gene/DATA/ (ncbi's ftp server)
# https://github.com/dhimmel/entrez-gene/blob/a7362748a34211e5df6f2d185bb3246279760546/download/Homo_sapiens.gene_info.gz <-- use pandas and trim i guess
gene_desc = pd.read_table("gene_desc.tsv")


def LF_CHECK_GENE_TAG(c):
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
            return 1
        elif token in gene_entry_df["description"].values[0].lower():
            return 1
    return -1


disease_desc = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv")
disease_normalization_df = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/slim-terms-prop.tsv")
wordnet_lemmatizer = WordNetLemmatizer()

def LF_CHECK_DISEASE_TAG(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    sen = c[0].get_parent()
    disease_name = re.sub("\)", "", c[0].get_span())

    # If abbreviation skip since no means of easy resolution
    if len(disease_name) <=5 and disease_name.isupper():
        return 0
    
    disease_name = [wordnet_lemmatizer.lemmatize(word) for word in disease_name.split(" ")]
    disease_name = " ".join(list(map(lambda x: x[0], filter(lambda x: x[1] == 'NN', nltk.pos_tag(disease_name)))))

    disease_id = sen.entity_cids[c[0].get_word_start()]
    disease_entry_df = disease_desc.query("doid_code == @disease_id")
    result = disease_normalization_df[disease_normalization_df["subsumed_name"].str.contains(r'^{}'.format(disease_name.lower()))]

    # If no match then return -1
    if result.empty:
        return -1
    else:
        # If it can be normalized return 1 else -1
        slim_id = result['slim_id'].values[0]
        if slim_id == disease_id:
            return 1
        else:
            return -1

"""
SENTENCE PATTERN MATCHING
"""

biomarker_indicators = {
                    "useful( marker of| in predicting)","(modulates the )?express(ed|ion)( of| on)?", 
                    "(prognostic|tissue|tumor) marker", "high (level(s)? of|concentrations of)", 
                    "have fewer", "quantification of", "evaluation of", "overproduced and hypersecreted by",
                    "assess the presensece of", "stained postively for", "elevated levels of",
                    "overproduced", "prognostic factor", "characterized by a marked increase of"
                    }

direct_association = {
                    "prognostic (significance of|indicator for|cyosolic factor|parameter for|information for)", "involved in",
                    "associat(ion|ed) (with|between)", "is an important independent variable",
                    "stimulated by", "high(er)? risk( for| of)?", "predictor of( prognosis in)?",
                    "corerlat(ion|ed) (between|with)", "significant( ultradian variation)?",
                    "showed that", "elevated risk for", "found in", "involved in","central role in",
                    "inhibited by", "greater for"                    
                    }
no_direct_association = {
                    "not(t)? significant(ly)?", "no(t)? assocait(ion|es)?",
                    "no correlat(ion|ed) (between|in|with)", "not detected in", "not been observed",
                    "not appear to be related to", "neither"
                    }
negative_indication =  {
    "failed", "poorly"
}

no_conclusion = {
    "investigated (the effect of|in)", "(was assessed by|assessed)", 
    "compared with", "were analyzed", "evaluated in", "examin(ation of|ed in)",
    "quantifi(ed in|cation by)", "we review", "(was|we) (measured| studied)( in)?",
    "derived from", "Regulation of", "are discussed", "to measure", "to study",
    "to explore", "detection of", "authors summarize", "responsiveness of",
    "used alone"}

def LF_IS_BIOMARKER(c):
    """
    This label function examines a sentences to determine of a sentence
    is talking about a biomarker. (A biomarker leads towards D-G assocation
    c - The candidate obejct being passed in
    """
    if re.search(ltp(biomarker_indicators), get_tagged_text(c)):
        return 1
    else:
        return 0

def LF_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is an association.
    """
    if re.search(ltp(direct_association), get_tagged_text(c)):
        return 1
    else:
        return 0


def LF_NO_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is no an association.
    """
    if re.search(ltp(no_direct_association), get_tagged_text(c)):
        return -1
    else:
        return 0

def LF_NO_CONCLUSION(c):
    if re.search(ltp(no_conclusion), get_tagged_text(c)):
        return -1
    else:
        return 0

def LF_DG_DISTANCE(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't right next to each other.
    """
    token_in_btw_len = len(get_text_between(c).split(" "))
    return -1 if token_in_btw_len < 5 else 0

def LF_NO_VERB(c):
    return -1 if len([x for x in  nltk.pos_tag(word_tokenize(c.get_parent().text)) if "VB" in x[1]])== 0 else 0

"""
DISEASE or GENE INDICATORS
"""

disease_prefix_indicators = {"metastasis of", "human", "severe", "in {{A}}",
                             "patients (presenting )?with", "primary", "diagnosis of",
                             "individuals with", "pathogenesis of", "cases of",
                             "positive (acute)?", "acute", "malignant", "de novo",
                             "levels in", "using"
                             }

disease_suffix_indiciators = {
                            "patients", "cell( lines|s)",
                            "disease", "subjects",
                            "inbreeding", "virus"
                            }

non_disease__indicators = {"enzymatic"}

gene_prefix_indicators = {
                        "mutations (in|of)", "recombinant",
                        "translocation", "binding assays",
                        "serum", "levels (of|in)", "variant( of|-like)",
                        "association of", "plasma", "increased", "concentration of"
                        }

gene_suffix_indicators = {
                        "(onco)?gene", "mRNA( transcript)?",
                        "translocation", "activity", "level(s)?",
                        "secretion", "immunoreactivity", "deficiency",
                        "lipoprotein", "test", "release", "antigens",
                        "antibodies"
                        }

non_gene_prefix_indicators = {"mixed leukocyte culture", "drugs"}
non_gene_suffix_indicators = {
    "cells", "lymphocytes",
    "system", ", [\d]+",
    "fibroblasts", "year(s)?",
    "Mustela vison"
}


def LF_GENE_PREFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a gene. It looks for key phrases/words that will
    suggest the possibility of the tagged entity being a gene
    """
    return 1 if re.search(ltp(gene_prefix_indicators), " ".join(get_left_tokens(c[1], window=5)), re.I) else 0


def LF_GENE_PREFIX_COMPLIMENT(c):
    """
    This LF is designed to find indicators that specify if the tagged gene is not a gene
    """
    return -1 if re.search(ltp(non_gene_prefix_indicators), " ".join(get_left_tokens(c[1], window=5)), re.I) else 0


def LF_GENE_SUFFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a gene. It looks for key phrases/words that will
    suggest the possibility of the tagged entity being a gene
    """
    return 1 if re.search(ltp(gene_suffix_indicators), " ".join(get_right_tokens(c[1], window=5)), re.I) else 0


def LF_GENE_SUFFIX_COMPLIMENT(c):
    """
    This LF is designed to find indicators that specify if the tagged gene is not a gene
    """
    return -1 if re.search(ltp(non_gene_suffix_indicators), " ".join(get_right_tokens(c[0], window=5)), re.I) else 0


def LF_DISEASE_PREFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a disease. It looks for key phrases/words that will
    suggest the possibility of the tagged entity being a disease.
    """
    return 1 if re.search(ltp(disease_prefix_indicators), " ".join(get_left_tokens(c[0], window=5)), re.I) else 0


def LF_DISEASE_SUFFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a disease. It looks for key phrases/words that will
    suggest the possibility of the tagged entity being a disease.
    """
    return 1 if re.search(ltp(disease_suffix_indiciators), " ".join(get_right_tokens(c[0], window=5)), re.I) else 0

def LF_NOISE(c):
    return -1


"""
RETRUN LFs to Notebook
"""


def get_lfs():
    """
    This helper function returns a list of each label function that
    will be used for the labeling step.
    """
    return [
            LF_HETNETS, LF_CHECK_GENE_TAG, LF_CHECK_DISEASE_TAG,
            LF_IS_BIOMARKER,LF_ASSOCIATION,
            LF_NO_ASSOCIATION,
            LF_NO_CONCLUSION,
            LF_DG_DISTANCE, LF_NO_VERB,
            LF_NOISE
            #LF_GENE_PREFIX,
            #LF_GENE_PREFIX_COMPLIMENT,
            #LF_GENE_SUFFIX, LF_GENE_SUFFIX_COMPLIMENT,
            #LF_DISEASE_PREFIX,
            #LF_DISEASE_SUFFIX

            ]
