import re

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


def LF_between_tag(c):
    m = re.search("associated with|Disruption", get_text_between(c))
    return 1 if m else 0


def LF_mutation(c):
    m = re.search("mutation", ",".join(get_left_tokens(c)))
    n = re.search("mutation", ",".join(get_right_tokens(c)))
    return 1 if m or n else 0


def LF_check_disease_tag(c):
    disease_name = c[0].get_span()
    if "syndrome" in disease_name:
        if "epilepsy" in disease_name.replace("syndrome", ""):
            return 1
        else:
            return -1
    else:
        return 1 if "epilepsy" in disease_name else 0
