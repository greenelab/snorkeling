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
import pathlib
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

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

def LF_HETNET_ABSENT(c):
    return 0 if any([
        LF_HETNET_DISEASES(c),
        LF_HETNET_DOAF(c),
        LF_HETNET_DisGeNET(c),
        LF_HETNET_GWAS(c)
    ]) else -1


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
            return 0
        elif token in gene_entry_df["description"].values[0].lower():
            return 0
    return -1


#disease_desc = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/xrefs-prop-slim.tsv")
disease_normalization_df = pd.read_table("https://raw.githubusercontent.com/dhimmel/disease-ontology/052ffcc960f5897a0575f5feff904ca84b7d2c1d/data/slim-terms-prop.tsv")
wordnet_lemmatizer = WordNetLemmatizer()

def LF_CHECK_DISEASE_TAG(c):
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
    "useful marker of", "useful in predicting","modulates the expression of", "expressed in" 
    "prognostic marker", "tissue marker", "tumor marker", "high level(s)? of", 
    "high concentrations of", "cytoplamsic concentration of",
    "have fewer", "quantification of", "evaluation of", "overproduced and hypersecreted by",
    "assess the presensece of", "stained postively for", "elevated levels of",
    "overproduced", "prognostic factor", "characterized by a marked increase of",
    "plasma levels of", "had elevated", "were detected", "was positive for"
    }

direct_association = {
    "prognostic significance of", "prognostic indicator for", "prognostic cyosolic factor",
    "prognostic parameter for", "prognostic information for", "involved in",
    "association with", "association between", "associated with", "associated between", 
    "is an important independent variable", "stimulated by", "high risk for", "higher risk of", "high risk for", "high risk of",
    "predictor of" "predictor of prognosis in", "correlation with", "correlation between", "correlated with",
    "correlated between", "significant ultradian variation", "significantly reduced",
    "showed that", "elevated risk for", "found in", "involved in","central role in",
    "inhibited by", "greater for", "indicative of" , "significantly reduced", "increased production of",
    "control the extent of", "secreted by"
    }

no_direct_association = {
    "not significant", "not significantly", "no association", "not associated",
    "no correlation between" "no correlation in", "no correlation with", "not correlated with",
     "not detected in", "not been observed", "not appear to be related to", "neither", 
     "provide evidence against"
    }

negative_indication =  {
    "failed", "poorly"
}

method_indication = {
    "investigated the effect of", "investigated in", "was assessed by", "assessed", 
    "compared with", "compared to", "were analyzed", "evaluated in", "examination of", "examined in",
    "quantified in" "quantification by", "we review", "was measured" "we studied", 
    "we measured", "derived from", "Regulation of", "are discussed", "to measure", "to study",
    "to explore", "detection of", "authors summarize", "responsiveness of",
    "used alone", "effect (of|on)", "blunting of", "measurement of", 
    "detection of", "occurence of", "response of", "stimulation by", 
    "our objective was", "to test the hypothesis", "studied in", "were reviewed",
    "randomized study", "this report considers"
    }

def LF_IS_BIOMARKER(c):
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

def LF_ASSOCIATION(c):
    """
    This LF is designed to test if there is a key phrase that suggests
    a d-g pair is an association.
    """
    if re.search(ltp(direct_association), get_text_between(c), flags=re.I):
        return 1
    elif re.search(ltp(direct_association) + r".*({{B}}|{{A}})", get_tagged_text(c), flags=re.I):
        return 1
    elif re.search(r"({{B}}|{{A}}).*" + ltp(direct_association), get_tagged_text(c), flags=re.I):
        return 1
    else:
        return 0

def LF_NO_ASSOCIATION(c):
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

def LF_METHOD_DESC(c):
    if re.search(ltp(method_indication), get_tagged_text(c), flags=re.I):
        return -1
    else:
        return 0

def LF_NO_CONCLUSION(c):
    return 0 if any([
        LF_ASSOCIATION(c),
        LF_IS_BIOMARKER(c),
        LF_NO_ASSOCIATION(c)
    ]) else -1

def LF_CONCLUSION(c):
    return 1 if any([
        LF_ASSOCIATION(c),
        LF_IS_BIOMARKER(c),
        LF_NO_ASSOCIATION(c)
    ]) else 0

def LF_DG_DISTANCE_SHORT(c):
    """
    This LF is designed to make sure that the disease mention
    and the gene mention aren't right next to each other.
    """
    return -1 if len(get_text_between(c).split(" ")) <=2 else 0

def LF_DG_DISTANCE_LONG(c):
    return -1 if len(get_text_between(c).split(" ")) > 50 else 0

def LF_DG_ALLOWED_DISTANCE(c):
    return 0 if any([
        LF_DG_DISTANCE_LONG(c),
        LF_DG_DISTANCE_SHORT(c)
        ]) else 1

def LF_NO_VERB(c):
    return -1 if len([x for x in  nltk.pos_tag(word_tokenize(c.get_parent().text)) if "VB" in x[1]])== 0 else 0


"""
RETRUN LFs to Notebook
"""

LFS = {
    "LF_HETNET_DISEASES": LF_HETNET_DISEASES,
    "LF_HETNET_DOAF": LF_HETNET_DOAF,
    "LF_HETNET_DisGeNET": LF_HETNET_DisGeNET,
    "LF_HETNET_GWAS": LF_HETNET_GWAS,
    "LF_HETNET_ABSENT":LF_HETNET_ABSENT,
    "LF_CHECK_GENE_TAG": LF_CHECK_GENE_TAG, 
    "LF_CHECK_DISEASE_TAG": LF_CHECK_DISEASE_TAG,
    "LF_IS_BIOMARKER": LF_IS_BIOMARKER,
    "LF_ASSOCIATION": LF_ASSOCIATION,
    "LF_NO_ASSOCIATION": LF_NO_ASSOCIATION,
    "LF_METHOD_DESC": LF_METHOD_DESC,
    "LF_NO_CONCLUSION": LF_NO_CONCLUSION,
    "LF_CONCLUSION": LF_CONCLUSION,
    "LF_DG_DISTANCE_SHORT": LF_DG_DISTANCE_SHORT,
    "LF_DG_DISTANCE_LONG": LF_DG_DISTANCE_LONG,
    "LF_DG_ALLOWED_DISTANCE": LF_DG_ALLOWED_DISTANCE,
    "LF_NO_VERB": LF_NO_VERB,
}
