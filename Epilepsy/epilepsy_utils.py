import cPickle

from collections import defaultdict
from itertools import product
from pandas import DataFrame
from string import punctuation

#Ripped off from the snorkel custom tagger.
#Designed to get the offset where a given token is found so it can receive a custom entity tag (i.e. Gene or Chemical)
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



#This is a custom class that is designed to tag each relevant word in a given sentence
# i.e if it see GAD then this tagger will give it a Gene tag.
class Tagger(object):
    def __init__(self,file_name):
        self.open_file = open(file_name,"r")

    #doing this because not guarenteed to be feed a sorted document list
    def retrieve_document(self,pubmed_id):
        for line in self.open_file:
            file_descriptor = line.split("::")
            if file_descriptor[0] == pubmed_id:
                document,tags = line.strip("\n").split("::")
                break
        self.open_file.seek(0)
        return [tuple([x if "|" in x else int(x) for x in data.split(",")]) for data in tags.split(":") if len(data.split(",")) == 3]


    def tag(self, parts):
        pubmed_id, _, _, sent_start, sent_end = parts['stable_id'].split(':')
        sent_start, sent_end = int(sent_start), int(sent_end)
        tags = self.retrieve_document(pubmed_id)

        #IGNORE for debugging purposes only
        #from IPython.core.debugger import Tracer
        #Tracer()() #this one triggers the debugger
        #print tags

        #For each tag in the given document
        #assign it to the correct word and move on
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
