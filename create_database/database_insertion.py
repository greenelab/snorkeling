from snorkel.parser import DocPreprocessor

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

    def __init__(self, filter_df):
        """ Initialize the tagger class
        Keyword arguments:
        self -- the class object
        filter_df -- a pandas group object for quick indexing
        """
        self.annt_df = filter_df

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
        # if int(pubmed_id) not in self.annt_df['pubmed_id']:
        #    return parts
        try:
            for index, tag in self.annt_df.get_group(int(pubmed_id)).iterrows():
                if not (sent_start <= int(tag['offset']) <= sent_end):
                    continue

                offsets = [offset + sent_start for offset in parts['char_offsets']]
                toks = offsets_to_token(int(tag['offset']), int(tag['end']), offsets, parts['lemmas'])
                for tok in toks:
                    parts['entity_types'][tok] = tag['type']
                    parts['entity_cids'][tok] = tag['identifier']

            return parts

        except KeyError as e:
            return parts
        
class XMLMultiDocPreprocessor(DocPreprocessor):
    """Hijacked this class to make it memory efficient
        Fixes the main issue where crashes if GB files are introduced
        et.iterparse for the win in memory efficiency
    """
    def __init__(self, path, doc='.//document', text='./text/text()', id='./id/text()', tag_filter=None):
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
        self.tag_filter = tag_filter

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

            if int(doc_id) not in self.tag_filter:
                doc.clear()
                continue

            text = '\n'.join(filter(lambda t: t is not None, doc.xpath(self.text)))

            # guarentees that resources are freed after they have been used
            doc.clear()
            meta = {'file_name': str(file_name)}
            stable_id = self.get_stable_id(doc_id)
            if not(text):
                continue

            yield Document(name=doc_id, stable_id=stable_id, meta=meta), text

    def _can_read(self, fpath):
        """ Straight forward function
        Keyword Arguments:
        fpath - the absolute path of the file.
        """
        return fpath.endswith('.xml')

def insert_cand_to_db(extractor, sentences):
    for split, sens in enumerate(sentences):
        extractor.apply(sens, split=split, parallelism=5, clear=False)
    
def print_candidates(session, context_class, edge):
    for i, label in enumerate(["Train", "Dev", "Test"]):
        cand_len = session.query(context_class).filter(context_class.split == i).count()
        print("Number of Candidates for {} edge and {} set: {}".format(edge, label, cand_len))