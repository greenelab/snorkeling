import cPickle

from collections import defaultdict
from itertools import product
from pandas import DataFrame
from string import punctuation

def offsets_to_token(left, right, offset_array, lemmas, punc=set(punctuation)):
    token_start, token_end = None, None
    for i, c in enumerate(offset_array):
        if left >= c:
            token_start = i
        if c > right and token_end is None:
            token_end = i
            break
    token_end = len(offset_array) - 1 if token_end is None else token_end
    token_end = token_end - 1 if lemmas[token_end - 1] in punc else token_end
    return range(token_start, token_end)

class Tagger(object):
    
    tag_dict = cPickle.load(open('data/Epilepsy/epilepsy_tags.pkl', 'rb'))

    def tag(self, parts):
        pubmed_id, _, _, sent_start, sent_end = parts['stable_id'].split(':')
        sent_start, sent_end = int(sent_start), int(sent_end)
        tags = self.tag_dict.get(pubmed_id, {})
        #from IPython.core.debugger import Tracer
        #Tracer()() #this one triggers the debugger
        #print tags
        for tag in tags:
            if not (sent_start <= tag[1] <= sent_end):
                continue
            offsets = [offset + sent_start for offset in parts['char_offsets']]
            toks = offsets_to_token(tag[1], tag[2], offsets, parts['lemmas'])
            for tok in toks:
                ts = tag[0].split('|')
                parts['entity_types'][tok] = ts[0]
                parts['entity_cids'][tok] = ts[1]
        return parts
