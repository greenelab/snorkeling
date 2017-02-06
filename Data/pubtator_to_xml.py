import argparse
import gzip
import cgi
import csv
import time
import sys

from __future__ import unicode_literals
from bioc import BioCWriter, BioCCollection, BioCDocument, BioCPassage
from bioc import BioCAnnotation, BioCLocation
from lxml.builder import E
from lxml.etree import tostring
import tqdm


def bioconcepts2pubtator_annotations(tag, index):
    """Bioconcepts to Annotations
    Specifically for bioconcepts2pubtator and converts each annotation
    into an annotation object that BioC can parse.

    Keyword Arguments:
    tag -- the annotation line that was parsed into an array
    index -- the id of each document specific annotation
    """

    annt = BioCAnnotation()
    annt.id = str(index)
    annt.infons["type"] = tag[4]

    # If the annotation type is a Gene,Species, Mutation, SNP
    # Write out relevant tag
    if tag[4] == "Gene":
        annt.infons["NCBI Gene"] = tag[5]

    elif tag[4] == "Species":
        annt.infons["NCBI Species"] = tag[5]

    elif "Mutation" in tag[4]:
        annt.infons["tmVar"] = tag[5]

    elif "SNP" in tag[4]:
        annt.infons["tmVar"] = tag[5]

    else:
        # If there is no MESH ID for an annotation
        if len(tag) > 5:
            # check to see if there are multiple mesh tags
            if "|" in tag[5]:
                # Write out each MESH id as own tag
                for tag_num, ids in enumerate(tag[5].split("|")):
                    # Some ids dont have the MESH:#### form so added case to that
                    if ":" not in ids:
                        annt.infons["MESH {}".format(tag_num)] = tag[5]
                    else:
                        term_type, term_id = ids.split(":")
                        annt.infons["{} {}".format(term_type, tag_num)] = term_id
            else:
                # Some ids dont have the MESH:#### form so added case to that
                if ":" in tag[5]:
                    term_type, term_id = tag[5].split(":")
                    annt.infons[term_type] = term_id
                else:
                    annt.infons["MESH"] = tag[5]
        else:
            annt.infons["MESH"] = "Unknown"

    location = BioCLocation()
    location.offset = str(tag[1])
    location.length = str(len(tag[3]))
    annt.locations.append(location)
    annt.text = tag[3]
    return annt


def bioconcepts2pubtator_offsets(input_file):
    file_lines = list()
    with gzip.open(input_file, "rb") as f:
        for line in f:
            # Convert "illegal chracters" (i.e. < > &) in the main text
            # into html entities
            line = cgi.escape(line.rstrip()).encode("ascii", "xmlcharrefreplace")
            if line:
                file_lines.append(line)
            else:
                article = {}

                # title
                title_heading = file_lines[0].split('|')
                title_len = len(title_heading[1])
                artilce["Title"] = [title_heading[0], title_heading[1]]

                # abstract
                abstract_heading = file_lines[1]
                article["Abstract"] = [abstract_heading[0], title_len, abstract_heading[1]]

                # set up the csv reader
                annotation_lines = "\n".join(file_lines[2:])
                annts = csv.DictReader(annotation_lines, fieldnames=['Document', 'Start', 'End', 'Term', 'Type', 'ID'])
                sorted_annts = sorted(annts, key=lambda x: x[2])
                article["Title_Annot"] = filter(lambda x: x[2] < title_len, sorted_annts)
                article["Abstract_Annot"] = filter(lambda x: x[2] > title_len, sorted_annts)

                yield article
                file_lines = list()


def convert_pubtator(input_file, output_file=None):
    """Convert pubtators annotation list to BioC XML

    Keyword Arguments:
    input_file -- the path of pubtators annotation file
    output_file -- the path to output the converted text
    """
    if output_file is None:
        output_file = "bioc-converted-docs.xml"

    # Set up BioCWriter to write specifically Pubtator
    # Can change to incorporate other sources besides pubtator
    writer = BioCWriter()
    writer.collection = BioCCollection()
    collection = writer.collection
    collection.date = time.strftime("%Y/%m/%d")
    collection.source = "Pubtator"
    collection.key = "Pubtator.key"

    with open(output_file, 'wb') as g:
        term_tags = set()

        # Have to manually do this because hangs otherwise
        # Write the head of the xml file
        xml_header = writer.tostring('UTF-8')
        xml_end = u'</collection>\n'
        xml_head = xml_header[:-len(xml_end)]
        g.write(xml_head)

        # Write each article in BioC format
        for article in bioconcepts2pubtator_offsets:
            document = BioCDocument()
            document.id = article["Title"][0]

            title_passage = BioCPassage()
            title_passage.put_infon('type', 'title')
            title_passage.offset = '0'
            title_passage.text = article["Title"][1]

            abstract_passage = BioCPassage()
            abstract_passage.put_infon('type', 'abstract')
            abstract_passage.offset = str(article["Abstract"][1])
            abstract_passage.text = article["Abstract"][2]

            for tag in article["Title_Annot"]:
                title_passage.annotations.append(get_annotations(tag, id_index))
                id_index = id_index + 1

            for tag in article["Abstract_Annot"]:
                abstract_passage.annotations.append(get_annotations(tag, id_index))
                id_index = id_index + 1

            document.add_passage(title_passage)
            document.add_passage(abstract_passage)

            step_parent = E('collection')
            writer._build_documents([document], step_parent)
            g.write(tostring(step_parent[0], pretty_print=True))
            step_parent.clear()

        # Write the closing tag of the xml document
        g.write(xml_tail)

# Main
parser = argparse.ArgumentParser(description='Extracts the annotations from the BioC xml format')
parser.add_argument("--documents", nargs=1, help="File path pointing to input file.")
parser.add_argument("--output", nargs="?", help="File path for destination of output.")
args = parser.parse_args()

if not(args.documents):
    raise Exception("PLEASE GIVE FILE INPUT PATH")

convert_pubtator(args.documents, args.output)
