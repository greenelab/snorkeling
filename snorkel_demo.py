import sys,os
#since you cant install it as a package (no setup.py) use the next best thing
sys.path.insert(1,'/home/davidnicholson/Documents/snorkel')
os.environ['SNORKELHOME'] = '/home/davidnicholson/Documents/snorkel'

from snorkel import SnorkelSession
from snorkel.parser import TSVDocParser,SentenceParser,CorpusParser
session = SnorkelSession()
parser = TSVDocParser(path="/home/davidnicholson/Documents/snorkel/tutorials/intro/data/articles-train.tsv")

#set up the sentence parser
sent_parser = SentenceParser()

#set up the corpus parser
cp = CorpusParser(parser,sent_parser)
corpus = cp.parse_corpus(session,'News Training')