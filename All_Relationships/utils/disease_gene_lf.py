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

hetnet_kb = pd.read_csv("hetnet_dg_kb.csv")


def ltp(tokens):
    return '(' + '|'.join(tokens) + ')'

def LF_DEBUG(c):
    """
    This label function is for debugging purposes. Feel free to ignore.
    keyword arguments:
    c - The candidate object to be labeled
    """
    print c
    print
    print "Left Tokens"
    print list(get_left_tokens(c[0], window=5))
    print
    print "Right Tokens"
    print list(get_right_tokens(c[0]))
    print
    print "Between Tokens"
    print list(get_between_tokens(c))
    print 
    print "Tagged Text"
    print get_tagged_text(c)
    print re.search(r'{{B}} .* is a .* {{A}}', get_tagged_text(c))
    print
    print "Get between Text"
    print get_text_between(c)
    print len(get_text_between(c))
    print 
    print "Parent Text"
    print c.get_parent()
    print
    return 0


def LF_IN_KB(c):
    """
    This label function is used for labeling each passed candidate as either pos or neg.
    Keyword Args:
    c- the candidate object to be passed in.
    """
    if not hetnet_kb[(hetnet_kb["disease_id"] == str(c.Disease_cid)) & (hetnet_kb["gene_id"] == int(c.Gene_cid))].empty:
        return 1
    else:
        return -1

biomarker_indicators = ["is reduced", "elevated in", "(excessive)? deposition of",
                        "high density", "accumulates in", "higher in", "lower in",
                        "correlate with", "measured by", "increased in", "decreased in",
                        "levels in"]


def LF_IS_BIOMARKER(c):
    """
    This label function examines a sentences to determine of a sentence
    is talking about a biomarker. (A biomarker leads towards D-G assocation
    c - The candidate obejct being passed in
    """
    return rule_regex_search_btw_AB(c, ltp(biomarker_indicators), 1) or rule_regex_search_btw_BA(c, ltp(biomarker_indicators), 1)


direct_association = ["associat(ion|ed) with", "central role in", "found in", "express(ion|ed)", "caus(ing|es|ed)", "observed in"]


def LF_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is an association.
    """
    return 1 if re.search(ltp(direct_association), get_text_between(candidate), re.I) return 0

direct_assocation_complement = ["not(t)? significant(ly)?", "no assocait(ion|es)?"]


def LF_NO_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is no an association.
    """
    return -1 if re.search(ltp(direct_assocation_complement), get_text_between(candidate), re.I) return 0

gene_prefix_indicators = ["mutations (in|of)", "recombinant", "translocation", "binding assays"]


def LF_GENE_PREFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a gene. It looks for key phrases/words that will 
    suggest the possibility of the tagged entity being a gene
    """
    return rule_regex_search_before_B(c, ltp(gene_prefix_indicators), 1)

gene_suffix_indicators = ["(onco)?gene", "mRNA (transcript)?", "translocation"]


def LF_GENE_SUFFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a gene. It looks for key phrases/words that will 
    suggest the possibility of the tagged entity being a gene
    """
    return 1 if re.search(r'{{B}} ' + ltp(gene_suffix_indicators), get_tagged_text(c) re.I) else 0

disease_prefix_indicators = ["patients with", "diagnosis of", "individuals with", "pathogenesis of", "cases of"]


def LF_DISEASE_PREFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a disease. It looks for key phrases/words that will 
    suggest the possibility of the tagged entity being a disease.
    """
    return rule_regex_search_before_A(c, ltp(gene_prefix_indicators), 1)

disease_suffix_indiciators = ["patients", "cells"]


def LF_DISEASE_SUFFIX(c):
    """
    This LF is designed to confirm that the entity labeld as gene
    is really a disease. It looks for key phrases/words that will 
    suggest the possibility of the tagged entity being a disease.
    """
    return 1 if re.search(r'{{A}} ' + ltp(gene_suffix_indicators), get_tagged_text(c) re.I) else 0


def get_lfs():
    """
    This helper function returns a list of each label function that will be used
    for the labeling step.
    """
    return [
            LF_IN_KB, LF_ASSOCIATION,
            LF_NO_ASSOCIATION, LF_IS_BIOMARKER,
            LF_DISEASE_SUFFIX, LF_DISEASE_PREFIX,
            LF_GENE_PREFIX, LF_GENE_SUFFIX
            ]
