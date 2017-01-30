from collections import defaultdict
from itertools import product
from string import punctuation

from pandas import DataFrame

def offsets_to_token(left, right, offset_array, lemmas, punc=set(punctuation)):
    """Calculate the offset from tag to token

    Ripped off from the snorkel custom tagger.
    Designed to get the offset where a given token is found so it can receive a custom entity tag.
    (i.e. Gene or Chemical)

    Keyword arguments
    left - the start of the tag
    right - the end of the tag
    offset_array - array of offsets given by stanford corenlp 
    lemmas - array of lemmas given by stanford corenlp
    punc - a list of punctuation characters
    """
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
    """Custom Tagger Class
    This is a custom class that is designed to tag each relevant word in a given sentence
    i.e if it see GAD then this tagger will give GAD a Gene tag.
    """
    
    def __init__(self,file_name):
        """ Initialize the tagger class

        Keyword arguments:
        self -- the class object
        file_name -- the name of the file that contains the document annotations
        """
        self.open_file = open(file_name,"r")

    
    def __del__(self):
        self.open_file.close()

    def retrieve_document(self,pubmed_id):
        """Retrieve pubtator's annotations and pythonize them.
        
        Keyword arguments:
        self -- The class object
        pubmed_id -- The id of the pubmed abstract

        Returns:
        A python object containing the annotations specific to a given document
        """

        #perform inline search to find correct document
        for line in self.open_file:
            file_descriptor, tags = line.rstrip('\r\n').split("::")
            if file_descriptor[0] == pubmed_id:
                break
        #Set file pointer back to beginning of file 
        self.open_file.seek(0) 
        converted_tags = []

        #for each document annotation pythonize it
        for data in tags.split(":"):
            tagged_offsets = data.split(",")
            if len(tagged_offsets) == 3:
                temp_tags = []
                for x in data.split(","):
                    if "|" in x:
                        temp_tags.append(x)
                    else:
                        temp_tags.append(int(x))
                converted_tags.append(tuple(temp_tags))

        return converted_tags

    
    def tag(self, parts):
        """Tag each Sentence

        Keyword arguments:
        self -- the class object
        parts -- standford's corenlp object which consists of nlp properties

        Returns:
        An updated parts object containing specified custom tags.
        """

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
