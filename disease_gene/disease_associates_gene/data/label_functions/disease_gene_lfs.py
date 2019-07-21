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
path = pathlib.Path(__file__).joinpath('../../disease_associates_gene.csv.xz').resolve()
pair_df = pd.read_csv(path, dtype={"sources": str})
knowledge_base = set()
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        key = str(row.entrez_gene_id), row.doid_id, source
        knowledge_base.add(key)
        
path = pathlib.Path(__file__).joinpath('../../../../disease_downregulates_gene/disease_downregulates_gene.tsv.xz').resolve()
pair_df = pd.read_table(path, dtype={"sources": str})
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        key = str(row.entrez_gene_id), row.doid_id, source+'_down'
        knowledge_base.add(key)        

path = pathlib.Path(__file__).joinpath('../../../../disease_upregulates_gene/disease_upregulates_gene.tsv.xz').resolve()
pair_df = pd.read_table(path, dtype={"sources": str})
for row in pair_df.itertuples():
    if not row.sources or pd.isnull(row.sources):
        continue
    for source in row.sources.split('|'):
        key = str(row.entrez_gene_id), row.doid_id, source+'_up'
        knowledge_base.add(key)

def LF_HETNET_DISEASES(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the Diseases database
    """
    return 1 if (c.Gene_cid, c.Disease_cid, "DISEASES") in knowledge_base else 0

def LF_HETNET_DOAF(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the DOAF database
    """
    return 1 if (c.Gene_cid, c.Disease_cid, "DOAF") in knowledge_base else 0

def LF_HETNET_DisGeNET(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the DisGeNET database
    """
    return 1 if (c.Gene_cid, c.Disease_cid, "DisGeNET") in knowledge_base else 0

def LF_HETNET_GWAS(c):
    """
    This label function returns 1 if the given Disease Gene pair is
    located in the GWAS database
    """
    return 1 if (c.Gene_cid, c.Disease_cid, "GWAS Catalog") in knowledge_base else 0

def LF_HETNET_STARGEO_UP(c):
    return 1 if (c.Gene_cid, c.Disease_cid, "strego_up") in knowledge_base else 0

def LF_HETNET_STARGEO_DOWN(c):
    return 1 if (c.Gene_cid, c.Disease_cid, "strego_down") in knowledge_base else 0

def LF_HETNET_DaG_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear 
    in the databases above.
    """
    return 0 if any([
        LF_HETNET_DISEASES(c),
        LF_HETNET_DOAF(c),
        LF_HETNET_DisGeNET(c),
        LF_HETNET_GWAS(c)
    ]) else -1

def LF_HETNET_DuG_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear 
    in the databases above.
    """
    return 0 if LF_HETNET_STARGEO_UP(c) else -1

def LF_HETNET_DdG_ABSENT(c):
    """
    This label function fires -1 if the given Disease Gene pair does not appear 
    in the databases above.
    """
    return 0 if LF_HETNET_STARGEO_DOWN(c) else -1

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
    disease_name = " ".join([word for word in word_tokenize(disease_name) if word not in set(stop_word_list)])

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
    "serum", "expressed on", "overexpression of", "plasma", "over-expression", "high expression"
    "detection marker", "increased", "was enhanced", "was elevated in", "expression (in|of)",
    "significantly higher (concentrations of|in)", "higher and lower amounts of", "measurement of",
    "levels discriminate", "potential biomarker of", "elevated serum levels", "elevated"
    }

cellular_activity = {
    "positive immunostaining", "stronger immunoreactivity",
    "in vitro kinase activity", "incude proliferation", "apoptosis in",
    "early activation", "activation in", "depletion inhibited", "transcriptional activity",
    "transcriptionally activates","anti-tumor cell efficacy", "suppresses the development and progression",
    "secret an adaquate amount of", "epigenetic alteration of", "actively transcribe",
    "decreased {{B}} production in", "rna targeting {{B}}", "suppresses growth of human",
    "inhibits growth of", "partial agonist", "mediates {{B}} pi3k signaling",
    "induces apoptosis", "antitumor activity of", "{{B}} stained", "(?<!not ){{B}} agonist(s)?",
    "produc(e|tion).*{{B}}", "sensitizes {{A}}", "endocannabinoid involvement", "epigenetically regulates",
    "actively transcribe", "re-expression of {{B}}"
}
direct_association = {
    "association (with|of)", "association between", "associated with", "associated between", 
    "stimulated by", "correlat(ed|es|ion)? between", "correlat(e|ed|es|ion)? with",
    "significant ultradian variation", "showed (that|loss)", "found in", "involved in","central role in",
    "inhibited by", "greater for", "indicative of","increased production of",
    "control the extent of", "secreted by", "detected in", "positive for", "to be mediated", 
    "was produced by", "stimulates", "precipitated by", "affects", "counteract cholinergic deficits", 
    "mediator of", "candidate gene", "categorized", "positive correlation", 
    "regulated by", "important role in", "significant amounts of", "to contain", 
    "increased risk of", "express", "susceptibility gene(s)? for", "risk factor for", 
    "necessary and sufficient to", "associated gene", "plays crucial role in",
    "common cause of", "discriminate", "were observed"
    }
upregulates = {
    r"\bhigh\b", "elevate(d|s)?", "greated for",
    "greater in", "higher", "prevent their degeneration", "gain", "increased",
    "positive", "strong", "elevated", "upregulated", "up-regulat(ed|ion)", "higher",
    "was enhanced", "over-expression", "overexpression", "phosphorylates", "activated by",
    "significantly higher concentrations of", "highly expressed in", "3-fold higher expression of"
}
downregulates = {
    r"\blow\b", "reduce(d|s)?", "(significant(ly)?)? decrease(d|s)?", "inhibited by", "not higher",
    "unresponsive", "under-expression", "underexpresed", "down-regulat(ed|ion)", "downregulated", "knockdown",
    "suppressed", "negative", "weak", "lower", "suppresses", "deletion of", "through decrease in",
}

disease_sample_indicators = {
    "tissue", "cell", "patient", "tumor", "cancer", "carcinoma",
    "cell line", "cell-line", "group", "blood", "sera", "serum", "fluid", "subset", 
    "case",
}

diagnosis_indicators = {
    "prognostic significance of", "prognostic indicator for", "prognostic cyosolic factor",
    "prognostic parameter for", "prognostic information for", "predict(or|ive) of",
    "predictor of prognosis in", "indicative of", "diagnosis of", "was positive for",
    "detection of", "determined by", "diagnositic sensitivity", "dianostic specificity",
    "prognostic factor", "variable for the identification", "potential therapeutic agent",
    "prognostic parameter for", "identification of", "psychophysiological index of suicdal risk",
    "reflects clinical activity"
}

no_direct_association = {
    "not significant", "not significantly", "no association", "not associated",
    "no correlation between" "no correlation in", "no correlation with", "not correlated with",
     "not detected in", "not been observed", "not appear to be related to", "neither", 
     "provide evidence against", "not a constant", "not predictive", "nor were they correlated with",
    "lack of", "correlation was lost in", "no obvious association", ", whereas", "do not support", 
    "not find an association", "little is known", "does( n't|n't) appear to affect", "no way to differentiate",
    "not predictor of", "roles are unknown", "independent of", "no expression of", "abscence of", 
    "are unknown", "not increased in", "not been elucidated"
    }

weak_association = {
    "not necessarily indicate", "the possibility", 
    "low correlation", "may be.*important", 
    "might facillitate","might be closely related to", 
    "has potential", "maybe a target for", "potential (bio)?marker for",
    "implicated in", "clinical activity in", "may represent", "mainly responsible for",
    "we hypothesized", "potential contributors", "suggests the diagnosis of", 
    "suspected of contributing"
}

method_indication = {
    "investigate(d)? (the effect of|in)?", "was assessed by", "assessed", 
    "compared to", "w(as|e|ere)? analy(z|s)ed", "evaluated in", "examination of", "examined in",
    "quantified in" "quantification by", "we review", "(were|was) measured", "we(re)?( have)? studied", 
    "we measured", "derived from", "(are|is) discussed", "to measure", "(prospective|to) study",
    "to explore", "detection of", "authors summarize", "responsiveness of",
    "used alone", "blunting of", "measurement of", "detection of", "occurence of", 
    "our objective( was)?", "to test the hypothesis", "studied in", "were reviewed",
    "randomized study", "this report considers", "was administered", "determinations of",
    "we examine(d)?", "(was|we|were|to) evaluate(d)?", "to establish", "were selected", "(authors|were|we) determined",
    "we investigated", "to assess", "analyses were done", "for the study of", r"^The effect of",
    "OBJECTIVE :", "PURPOSE :", "METHODS :", "were applied", "EXPERIMENTAL DESIGN :",
    "we explored", "the purpose of", "to understand how", "to examine",
    "was conducted", "to determine", "we validated", "we characterized",
    "aim of (our|this|the) (study|meta-analysis)", "developing a", "we tested for", " was demonstrate(d)?",
    "we describe", "were compared", "were categorized", "was studied", "we calculate(d)?",
    "sought to investigate", "this study aimed", "a study was made", "study sought"
}

title_indication = {
    "Effect of", "Evaluation of", 
    "Clincal value of", "Extraction of",
    "Responsiveness of", "The potential for",
    "as defined by immunohistochemistry", "Comparison between",
    "Characterization of", "A case of", "Occurrence of",
    "Inborn", "Episodic", "Detection of", "Immunostaining of",
    "Mutational analysis of", "Identification of", "souble expression of", 
    "expression of", "genetic determinants of", "prolactin levels in",
    "a study on", "association (of|between)", "analysis of"
}
genetic_abnormalities = {
    "deletions (in|of)", "mutation(s)? in", "polymorphism(s)?",
    "promoter variant(s)?", "recombinant human", "novel {{B}} gene mutation",
    "pleotropic effects on",
}

context_change_keywords = {
    ", but", ", whereas", 
    "; however,",
}

def LF_DG_IS_BIOMARKER(c):
    """
    This label function examines a sentences to determine of a sentence
    is talking about a biomarker. (A biomarker leads towards D-G assocation
    c - The candidate obejct being passed in
    """
    if LF_DG_METHOD_DESC(c) or LF_DG_TITLE(c):
        return 0
    elif re.search(ltp(biomarker_indicators), " ".join(get_left_tokens(c[1], window=10)), flags=re.I):
        return 1
    elif re.search(ltp(biomarker_indicators), " ".join(get_right_tokens(c[1], window=10)), flags=re.I):
        return 1
    else:
        return 0

def LF_DaG_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is an association.
    """
    left_window = " ".join(get_left_tokens(c[0], window=10)) + " ".join(get_left_tokens(c[1], window=10))
    right_window = " ".join(get_right_tokens(c[0], window=10)) + " ".join(get_right_tokens(c[1], window=10))
    found_negation = not re.search(r'\b(not|no)\b', left_window, flags=re.I)

    if LF_DG_METHOD_DESC(c) or LF_DG_TITLE(c):
        return 0
    elif re.search(r'(?<!not )(?<!no )' + ltp(direct_association), get_text_between(c), flags=re.I) and found_negation:
        return 1
    elif re.search(r'(?<!not )(?<!no )' + ltp(direct_association), left_window, flags=re.I) and found_negation:
        return 1
    elif re.search(r'(?<!not )(?<!no )' + ltp(direct_association), right_window, flags=re.I) and found_negation:
        return 1
    else:
        return 0

def LF_DaG_WEAK_ASSOCIATION(c):
    """
    This label function is design to search for phrases that indicate a 
    weak association between the disease and gene
    """
    left_window = " ".join(get_left_tokens(c[0], window=10)) + " ".join(get_left_tokens(c[1], window=10))
    right_window = " ".join(get_right_tokens(c[0], window=10)) + " ".join(get_right_tokens(c[1], window=10))
    
    if LF_DG_METHOD_DESC(c) or LF_DG_TITLE(c):
        return 0
    elif re.search(ltp(weak_association), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(weak_association), left_window, flags=re.I):
        return 1
    elif re.search(ltp(weak_association), right_window, flags=re.I):
        return 1
    else:
        return 0

def LF_DaG_NO_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is no an association.
    """
    left_window = " ".join(get_left_tokens(c[0], window=10)) + " ".join(get_left_tokens(c[1], window=10))
    right_window = " ".join(get_right_tokens(c[0], window=10)) + " ".join(get_right_tokens(c[1], window=10))
    
    if LF_DG_METHOD_DESC(c) or LF_DG_TITLE(c):
        return 0
    elif re.search(ltp(no_direct_association), get_text_between(c), flags=re.I):
        return -1
    elif re.search(ltp(no_direct_association), left_window, flags=re.I):
        return -1
    elif re.search(ltp(no_direct_association), right_window, flags=re.I):
        return -1
    else:
        return 0

def LF_DaG_CELLULAR_ACTIVITY(c):
    """
    This LF is designed to look for key phrases that indicate activity within a cell.
    e.x. positive immunostating for an experiment
    """
    left_window = " ".join(get_left_tokens(c[0], window=10)) + " ".join(get_left_tokens(c[1], window=10))
    right_window = " ".join(get_right_tokens(c[0], window=10)) + " ".join(get_right_tokens(c[1], window=10))
    
    if re.search(ltp(cellular_activity), get_tagged_text(c), flags=re.I):
        return 1
    elif re.search(ltp(cellular_activity), left_window, flags=re.I):
        return 1
    elif re.search(ltp(cellular_activity), right_window, flags=re.I):
        return 1
    else:
        return 0
    
def LF_DaG_DISEASE_SAMPLE(c):
    """
    This LF is designed to look for key phrases that indicate a sentence talking about tissue samples
    ex. cell line etc
    """
    left_window = " ".join(get_left_tokens(c[0], window=10)) + " ".join(get_left_tokens(c[1], window=10))
    right_window = " ".join(get_right_tokens(c[0], window=10)) + " ".join(get_right_tokens(c[1], window=10))
    
    if re.search(ltp(disease_sample_indicators), left_window, flags=re.I):
        return 1
    elif re.search(ltp(disease_sample_indicators), right_window, flags=re.I):
        return 1
    else:
        return 0
    
def LF_DG_METHOD_DESC(c):
    """
    This label function is designed to look for phrases 
    that imply a sentence is description an experimental design
    """
    sentence_tokens = " ".join(c.get_parent().words[0:20])
    if re.search(ltp(method_indication), sentence_tokens, flags=re.I):
        return -1
    elif re.search(ltp(method_indication), " ".join(get_between_tokens(c)), flags=re.I):
        return -1
    else:
        return 0

def LF_DG_TITLE(c):
    """
    This label function is designed to look for phrases that inditcates
    a paper title
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

def LF_DuG_UPREGULATES(c):
    """
    This label function is designed to search for words that indicate
    a sort of positive response or imply an upregulates association
    """
    if LF_DG_METHOD_DESC(c) or LF_DG_TITLE(c):
        return 0
    else:
        if rule_regex_search_btw_AB(c, r'.*'+ltp(upregulates)+r'.*', 1):
            return 1
        elif rule_regex_search_btw_BA(c, r'.*'+ltp(upregulates)+r'.*', 1):
            return 1
        elif re.search(r'({{A}}|{{B}}).*({{A}}|{{B}}).*' + ltp(upregulates), get_tagged_text(c)):
            return 1
        else:
            return 0

def LF_DdG_DOWNREGULATES(c):
    """
    This label function is designed to search for words that indicate
    a sort of negative response or imply an downregulates association
    """
    if LF_DG_METHOD_DESC(c) or LF_DG_TITLE(c):
        return 0
    else:
        if rule_regex_search_btw_AB(c, r'.*'+ltp(downregulates)+r'.*', 1):
            return 1
        elif rule_regex_search_btw_BA(c, r'.*'+ltp(downregulates)+r'.*', 1):
            return 1
        elif re.search(r'({{A}}|{{B}}).*({{A}}|{{B}}).*' + ltp(downregulates), get_tagged_text(c)):
            return 1
        else:
            return 0

def LF_DdG_METHYLATION(c):
    if "methylation" in get_tagged_text(c):
        return 1
    return 0

def LF_DG_GENETIC_ABNORMALITIES(c):
    """
    This LF searches for key phraes that indicate a genetic abnormality
    """
    left_window = " ".join(get_left_tokens(c[0], window=10)) + " ".join(get_left_tokens(c[1], window=10))
    right_window = " ".join(get_right_tokens(c[0], window=10)) + " ".join(get_right_tokens(c[1], window=10))
    
    if re.search(ltp(genetic_abnormalities), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(genetic_abnormalities), left_window, flags=re.I):
        return 1
    elif re.search(ltp(genetic_abnormalities), right_window, flags=re.I):
        return 1
    return 0
    
def LF_DG_DIAGNOSIS(c):
    """
    This label function is designed to search for words that imply a patient diagnosis
    which will provide evidence for possible disease gene association.
    """
    return 1 if any([rule_regex_search_btw_AB(c, r'.*'+ltp(diagnosis_indicators) + r".*", 1), rule_regex_search_btw_BA(c, r'.*'+ltp(diagnosis_indicators) + r".*", 1)]) or  \
        re.search(r'({{A}}|{{B}}).*({{A}}|{{B}}).*' + ltp(diagnosis_indicators), get_tagged_text(c)) else 0

def LF_DG_PATIENT_WITH(c):
    """
    This label function looks for the phrase "  with" disease.
    """
    return 1 if re.search(r"patient(s)? with.{1,200}{{A}}", get_tagged_text(c), flags=re.I) else 0

def LF_DG_CONCLUSION_TITLE(c):
    """"
    This label function searches for the word conclusion at the beginning of the sentence.
    Some abstracts are written in this format.
    """
    return 1 if "CONCLUSION:" in get_tagged_text(c) or "concluded" in get_tagged_text(c) else 0

def LF_DaG_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum([
        LF_DaG_ASSOCIATION(c), LF_DG_IS_BIOMARKER(c), LF_DG_DIAGNOSIS(c),
        LF_DaG_CELLULAR_ACTIVITY(c),
        np.abs(LF_DaG_WEAK_ASSOCIATION(c)), np.abs(LF_DaG_NO_ASSOCIATION(c))
    ])
    negative_num = np.abs(np.sum([
        LF_DG_METHOD_DESC(c), LF_DG_TITLE(c), 
        LF_DG_NO_VERB(c), LF_DG_MULTIPLE_ENTITIES(c)
    ]))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_DaG_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if LF_DaG_NO_ASSOCIATION(c) or LF_DaG_WEAK_ASSOCIATION(c):
        return -1
    elif not LF_DaG_NO_CONCLUSION(c):
        return 1
    else:
        return 0
    
def LF_DuG_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum([  
        LF_DuG_UPREGULATES(c)
    ])
    negative_num = np.abs(np.sum(LF_DG_METHOD_DESC(c), LF_DG_TITLE(c)))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_DuG_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_DuG_NO_CONCLUSION(c):
        return 1
    else:
        return 0
    
def LF_DdG_NO_CONCLUSION(c):
    """
    This label function fires a -1 if the number of negative label functinos is greater than the number
    of positive label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association.
    """
    positive_num = np.sum([ 
            LF_DdG_DOWNREGULATES(c)
    ])
    negative_num = np.abs(np.sum(LF_DG_METHOD_DESC(c), LF_DG_TITLE(c)))
    if positive_num - negative_num >= 1:
        return 0
    return -1

def LF_DdG_CONCLUSION(c):
    """
    This label function fires a 1 if the number of positive label functions is greater than the number
    of negative label functions.
    The main idea behind this label function is add support to sentences that could
    mention a possible disease gene association
    """
    if not LF_DdG_NO_CONCLUSION(c):
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
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't too far from each other.
    """
    return -1 if len(list(get_between_tokens(c))) > 50 else 0

def LF_DG_ALLOWED_DISTANCE(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention are in an acceptable distance between 
    each other
    """
    return 0 if any([
        LF_DG_DISTANCE_LONG(c),
        LF_DG_DISTANCE_SHORT(c)
        ]) else 1

def LF_DG_NO_VERB(c):
    """
    This label function is designed to fire if a given
    sentence doesn't contain a verb. Helps cut out some of the titles
    hidden in Pubtator abstracts
    """
    if len([x for x in  c.get_parent().pos_tags if "VB" in x and x != "VBG"]) == 0:
        return -1
    return 0

def LF_DG_MULTIPLE_ENTITIES(c):
    #Keep track of previous entity
    previous_entity = 'o'
    
    #Count the number of entities
    entity_count = 0
    
    for entity in c.get_parent().entity_types:
        entity = entity.lower()

        #Non-O tag
        if entity != 'o' and previous_entity =='o':
            entity_count += 1

        previous_entity = entity
        
    if entity_count > 2:
        return -1
    return 0


def LF_DG_CONTEXT_SWITCH(c):
    if re.search(ltp(context_change_keywords), get_text_between(c), flags=re.I):
        return -1
    return 0

"""
Bi-Clustering LFs
"""
path = pathlib.Path(__file__).joinpath("../../../../../dependency_cluster/disease_gene_bicluster_results.tsv.xz").resolve()
bicluster_dep_df = pd.read_table(path)
causal_mutations_base = set([tuple(x) for x in bicluster_dep_df.query("U>0")[["pubmed_id", "sentence_num"]].values])
mutations_base = set([tuple(x) for x in bicluster_dep_df.query("Ud>0")[["pubmed_id", "sentence_num"]].values])
drug_targets_base = set([tuple(x) for x in bicluster_dep_df.query("D>0")[["pubmed_id", "sentence_num"]].values])
pathogenesis_base = set([tuple(x) for x in bicluster_dep_df.query("J>0")[["pubmed_id", "sentence_num"]].values])
therapeutic_base = set([tuple(x) for x in bicluster_dep_df.query("Te>0")[["pubmed_id", "sentence_num"]].values])
polymorphisms_base = set([tuple(x) for x in bicluster_dep_df.query("Y>0")[["pubmed_id", "sentence_num"]].values])
progression_base = set([tuple(x) for x in bicluster_dep_df.query("G>0")[["pubmed_id", "sentence_num"]].values])
biomarkers_base = set([tuple(x) for x in bicluster_dep_df.query("Md>0")[["pubmed_id", "sentence_num"]].values])
overexpression_base = set([tuple(x) for x in bicluster_dep_df.query("X>0")[["pubmed_id", "sentence_num"]].values])
regulation_base = set([tuple(x) for x in bicluster_dep_df.query("L>0")[["pubmed_id", "sentence_num"]].values])

def LF_DG_BICLUSTER_CASUAL_MUTATIONS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in causal_mutations_base:
        return 1
    return 0

def LF_DG_BICLUSTER_MUTATIONS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in mutations_base:
        return 1
    return 0

def LF_DG_BICLUSTER_DRUG_TARGETS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in drug_targets_base:
        return 1
    return 0

def LF_DG_BICLUSTER_PATHOGENESIS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in pathogenesis_base:
        return 1
    return 0

def LF_DG_BICLUSTER_THERAPEUTIC(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in therapeutic_base:
        return 1
    return 0

def LF_DG_BICLUSTER_POLYMORPHISMS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in polymorphisms_base:
        return 1
    return 0

def LF_DG_BICLUSTER_PROGRESSION(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in progression_base:
        return 1
    return 0

def LF_DG_BICLUSTER_BIOMARKERS(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in biomarkers_base:
        return 1
    return 0

def LF_DG_BICLUSTER_OVEREXPRESSION(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in overexpression_base:
        return 1
    return 0

def LF_DG_BICLUSTER_REGULATION(c):
    """
    This label function uses the bicluster data located in the 
    A global network of biomedical relationships
    """
    sen_pos = c.get_parent().position
    pubmed_id = int(c.get_parent().document.name)
    if (pubmed_id, sen_pos) in regulation_base:
        return 1
    return 0

"""
RETRUN LFs to Notebook
"""

DG_LFS = {
    "DaG":
    OrderedDict({
        "LF_HETNET_DISEASES": LF_HETNET_DISEASES,
        "LF_HETNET_DOAF": LF_HETNET_DOAF,
        "LF_HETNET_DisGeNET": LF_HETNET_DisGeNET,
        "LF_HETNET_GWAS": LF_HETNET_GWAS,
        "LF_HETNET_DaG_ABSENT":LF_HETNET_DaG_ABSENT,
        "LF_DG_CHECK_GENE_TAG": LF_DG_CHECK_GENE_TAG, 
        "LF_DG_CHECK_DISEASE_TAG": LF_DG_CHECK_DISEASE_TAG,
        "LF_DG_IS_BIOMARKER": LF_DG_IS_BIOMARKER,
        "LF_DaG_ASSOCIATION": LF_DaG_ASSOCIATION,
        "LF_DaG_WEAK_ASSOCIATION": LF_DaG_WEAK_ASSOCIATION,
        "LF_DaG_NO_ASSOCIATION": LF_DaG_NO_ASSOCIATION,
        "LF_DaG_CELLULAR_ACTIVITY":LF_DaG_CELLULAR_ACTIVITY,
        "LF_DaG_DISEASE_SAMPLE":LF_DaG_DISEASE_SAMPLE,
        "LF_DG_METHOD_DESC": LF_DG_METHOD_DESC,
        "LF_DG_TITLE": LF_DG_TITLE,
        "LF_DG_GENETIC_ABNORMALITIES":LF_DG_GENETIC_ABNORMALITIES,
        "LF_DG_DIAGNOSIS": LF_DG_DIAGNOSIS,
        "LF_DG_PATIENT_WITH":LF_DG_PATIENT_WITH,
        "LF_DG_CONCLUSION_TITLE":LF_DG_CONCLUSION_TITLE,
        "LF_DaG_NO_CONCLUSION":LF_DaG_NO_CONCLUSION,
        "LF_DaG_CONCLUSION":LF_DaG_CONCLUSION,
        "LF_DG_DISTANCE_SHORT": LF_DG_DISTANCE_SHORT,
        "LF_DG_DISTANCE_LONG": LF_DG_DISTANCE_LONG,
        "LF_DG_ALLOWED_DISTANCE": LF_DG_ALLOWED_DISTANCE,
        "LF_DG_NO_VERB": LF_DG_NO_VERB,
        "LF_DG_MULTIPLE_ENTITIES":LF_DG_MULTIPLE_ENTITIES,
        "LF_DG_CONTEXT_SWITCH":LF_DG_CONTEXT_SWITCH,
        "LF_DG_BICLUSTER_CASUAL_MUTATIONS":LF_DG_BICLUSTER_CASUAL_MUTATIONS,
        "LF_DG_BICLUSTER_MUTATIONS":LF_DG_BICLUSTER_MUTATIONS,
        "LF_DG_BICLUSTER_DRUG_TARGETS":LF_DG_BICLUSTER_DRUG_TARGETS,
        "LF_DG_BICLUSTER_PATHOGENESIS":LF_DG_BICLUSTER_PATHOGENESIS,
        "LF_DG_BICLUSTER_THERAPEUTIC":LF_DG_BICLUSTER_THERAPEUTIC,
        "LF_DG_BICLUSTER_POLYMORPHISMS":LF_DG_BICLUSTER_POLYMORPHISMS,
        "LF_DG_BICLUSTER_PROGRESSION":LF_DG_BICLUSTER_PROGRESSION,
        "LF_DG_BICLUSTER_BIOMARKERS":LF_DG_BICLUSTER_BIOMARKERS,
        "LF_DG_BICLUSTER_OVEREXPRESSION":LF_DG_BICLUSTER_OVEREXPRESSION,
        "LF_DG_BICLUSTER_REGULATION":LF_DG_BICLUSTER_REGULATION,
    }),
    "DuG":
    OrderedDict({
        "LF_HETNET_STARGEO_UP":LF_HETNET_STARGEO_UP,
        "LF_HETNET_DuG_ABSENT":LF_HETNET_DuG_ABSENT,
        "LF_DG_CHECK_GENE_TAG": LF_DG_CHECK_GENE_TAG, 
        "LF_DG_CHECK_DISEASE_TAG": LF_DG_CHECK_DISEASE_TAG,
        "LF_DG_IS_BIOMARKER": LF_DG_IS_BIOMARKER,
        "LF_DG_METHOD_DESC": LF_DG_METHOD_DESC,
        "LF_DG_TITLE": LF_DG_TITLE,
        "LF_DuG_UPREGULATES": LF_DuG_UPREGULATES,
        "LF_DG_PATIENT_WITH":LF_DG_PATIENT_WITH,
        "LF_DG_CONCLUSION_TITLE":LF_DG_CONCLUSION_TITLE,
        "LF_DuG_NO_CONCLUSION":LF_DuG_NO_CONCLUSION,
        "LF_DuG_CONCLUSION":LF_DuG_CONCLUSION,
        "LF_DG_DISTANCE_SHORT": LF_DG_DISTANCE_SHORT,
        "LF_DG_DISTANCE_LONG": LF_DG_DISTANCE_LONG,
        "LF_DG_ALLOWED_DISTANCE": LF_DG_ALLOWED_DISTANCE,
        "LF_DG_NO_VERB": LF_DG_NO_VERB,
    }),
    "DdG":
    OrderedDict({
        "LF_HETNET_STARGEO_DOWN":LF_HETNET_STARGEO_DOWN,
        "LF_HETNET_DdG_ABSENT":LF_HETNET_DdG_ABSENT,
        "LF_DG_CHECK_GENE_TAG": LF_DG_CHECK_GENE_TAG, 
        "LF_DG_CHECK_DISEASE_TAG": LF_DG_CHECK_DISEASE_TAG,
        "LF_DG_IS_BIOMARKER": LF_DG_IS_BIOMARKER,
        "LF_DG_METHOD_DESC": LF_DG_METHOD_DESC,
        "LF_DG_TITLE": LF_DG_TITLE,
        "LF_DdG_DOWNREGULATES": LF_DdG_DOWNREGULATES,
        "LF_DdG_METHYLATION": LF_DdG_METHYLATION,
        "LF_DG_PATIENT_WITH":LF_DG_PATIENT_WITH,
        "LF_DG_CONCLUSION_TITLE":LF_DG_CONCLUSION_TITLE,
        "LF_DuG_NO_CONCLUSION":LF_DdG_NO_CONCLUSION,
        "LF_DdG_CONCLUSION":LF_DdG_CONCLUSION,
        "LF_DG_DISTANCE_SHORT": LF_DG_DISTANCE_SHORT,
        "LF_DG_DISTANCE_LONG": LF_DG_DISTANCE_LONG,
        "LF_DG_ALLOWED_DISTANCE": LF_DG_ALLOWED_DISTANCE,
        "LF_DG_NO_VERB": LF_DG_NO_VERB,
    })
}
