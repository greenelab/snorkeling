from snorkel.lf_helpers import (
    get_left_tokens,
    get_right_tokens,
    get_between_tokens,
    get_tagged_text,
    get_text_between,
    rule_regex_search_tagged_text,
    rule_regex_search_btw_AB,
    rule_regex_search_btw_BA,
    rule_regex_search_before_A,
    rule_regex_search_before_B,
)
import pandas as pd

hetnet_kb = pd.read_csv("hetnet_dg_kb.csv")


def LF_DEBUG(C):
    """
    This label function is for debugging purposes. Feel free to ignore.
    keyword arguments:
    c - The candidate object to be labeled
    """

    print "Left Tokens"
    print get_left_tokens(c,window=3)
    print
    print "Right Tokens"
    print get_right_tokens(c)
    print
    print "Between Tokens"
    print get_between_tokens(c)
    print 
    print "Tagged Text"
    print get_tagged_text(c)
    print re.search(r'{{B}} .* is a .* {{A}}',get_tagged_text(c))
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


def get_lfs(debug):
    """
    This helper function returns a list of each label function that will be used
    for the labeling step.
    """
    return [LF_IN_KB] if not debug else [LF_DEBUG]
