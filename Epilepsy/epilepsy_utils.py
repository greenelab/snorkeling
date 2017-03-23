from collections import defaultdict
import csv
import os
from itertools import product
import shelve
from string import punctuation
import sys

import lxml.etree as et
from snorkel.parser import DocPreprocessor
from snorkel.models import Document


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
    This is a custom class that is designed to tag each relevant word
    in a given sentence.
    i.e if it sees GAD then this tagger will give GAD a Gene tag.
    """

    def __init__(self, file_name):
        """ Initialize the tagger class

        Keyword arguments:
        self -- the class object
        file_name -- the name of the file that contains the document annotations
        """
        self.annt_dict = shelve.open(file_name)

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

        # For each tag in the given document
        # assign it to the correct word and move on
        if pubmed_id not in self.annt_dict:
            return parts

        for tag in self.annt_dict[pubmed_id]:
            if not (sent_start <= int(tag['Offset']) <= sent_end):
                continue
            offsets = [offset + sent_start for offset in parts['char_offsets']]
            toks = offsets_to_token(int(tag['Offset']), int(tag['End']), offsets, parts['lemmas'])
            for tok in toks:
                parts['entity_types'][tok] = tag['Type']
                parts['entity_cids'][tok] = tag['ID']
        return parts


class XMLMultiDocPreprocessor(DocPreprocessor):
    """Hijacked this class to make it memory efficient
        Fixes the main issue where crashes if GB files are introduced
        et.iterparse for the win in memory efficiency
    """
    def __init__(self, path, doc='.//document', text='./text/text()', id='./id/text()'):
        """Initialize the XMLMultiDocPreprocessor Class

        Keyword Arguments:
        path - the absolute path of the xml file
        doc - the xpath notation for document obejcts
        test - the xpath for grabbing all the text objects
        id - the xpath for grabbing all the id tags
        """
        DocPreprocessor.__init__(self, path)
        self.doc = doc
        self.text = text
        self.id = id

    def parse_file(self, f, file_name):
        """This method overrides the original method

        Keyword arguments:
        f - the file object to be parsed by lxml
        file_name - the name of the file used as metadata

        Yields:
        A document object in sqlalchemy format and the corresponding text
        that will be parsed by CoreNLP
        """
        for event, doc in et.iterparse(f, tag='document'):
            doc_id = str(doc.xpath(self.id)[0])
            text = '\n'.join(filter(lambda t: t is not None, doc.xpath(self.text)))
            # guarentees that resources are freed after they have been used
            doc.clear()
            meta = {'file_name': str(file_name)}
            stable_id = self.get_stable_id(doc_id)
            assert text
            yield Document(name=doc_id, stable_id=stable_id, meta=meta), text

    def _can_read(self, fpath):
        """ Straight forward function

        Keyword Arguments:
        fpath - the absolute path of the file.
        """
        return fpath.endswith('.xml')
