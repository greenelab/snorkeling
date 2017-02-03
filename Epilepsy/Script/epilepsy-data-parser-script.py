import os

from epilepsy_utils import Tagger
from snorkel import SnorkelSession
from snorkel.models import Document, Sentence, candidate_subclass
from snorkel.modes import PretaggedCandidateExtractor
from snorkel.parser import XMLMultiDocPreprocessor, CorpusParser
import tqdm

# set up the session
os.environ['SNORKELDB'] = 'sqlite:///pubmed.db'
session = SnorkelSession()

# create the parser
print "Parsing the Documents!!"
xml_parser = XMLMultiDocPreprocessor(
    path="../data/Epilepsy/pubmed.xml",
    doc='.//document',
    text='.//passage/text/tect()',
    id='.//id/text()'
)

tagger = Tagger()
corpus_parser = CorpusPArser(fn=tagger.tag)
corpus_parser.apply(list(xml_parser))


print "Documents: {}".format(session.query(Document).count())
print "Sentence: {}".format(session.query(Sentence).count())

# grab the sentences
ids = [doc.name for doc in session.query(Document)]
indicies = len(ids)
first = int(indicies/3)
second = first * 2
train_ids, dev_ids, test_ids = ids[0:first], ids[first:second], ids[second:]

train_sents(), dev_sents, test_sents = set(), set(), set()
docs = session.query(Document).all()
for doc in tqdm.tqdm(docs):
    for s in doc.sentences:
        if doc.name in train_ids:
            train_sents.add(s)
        elif doc.name in dev_ids:
            dev_sents.add(s)
        else:
            test_sents.add(s)

# Grab the candidates
DiseaseGene = candidate_subclass('DiseaseGene', ['Disease', 'Gene'])
ce = PretaggedCandidateExtractor(DiseaseGene, ['Disease', 'Gene'])
for k, sents in enumerate([train_sents, dev_sents, test_sents]):
    ce.apply(sents, split=k)
    print "Number of Candidates: {}".format(session.query(DiseaseGene).filter(DiseaseGene.split == k).count())

print "Done"
