import gzip
import cgi
import time
import sys
from lxml.builder import E
from lxml.etree import tostring

from bioc import BioCWriter, BioCCollection, BioCDocument, BioCPassage
from bioc import BioCAnnotation, BioCLocation
import tqdm


def get_annotations(tag, index):
    """Get Annotations

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


def convert_pubtator(input_file, output_file):
    """Convert pubtators annotation list to XML

    Keyword Arguments:
    input_file -- the path of pubtators annotation file
    output_file -- the path to output the converted text
    """
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
        shell = writer.tostring('UTF-8')
        tail = u'</collection>\n'
        head = shell[:-len(tail)]
        g.write(head)

        # read from gunzip file
        with gzip.open(input_file, "rb") as f:
            for line in tqdm.tqdm(f):
                # Convert "illegal chracters" (i.e. < > &) in the main text
                # into html entities
                line = cgi.escape(line.strip()).encode("ascii", "xmlcharreplace")

                # Title parsing
                if "|t|" in line:
                    title_heading = line.split("|")
                    document = BioCDocument()
                    document.id = title_heading[0]

                    title_passage = BioCPassage()
                    title_passage.put_infon('type', 'title')
                    title_passage.offset = '0'
                    title_passage.text = title_heading[2]

                # Abstract parsing
                elif "|a|" in line:
                    abstract_heading = line.split("|")
                    title_offset = len(title_heading[2])
                    abstract_passage = BioCPassage()
                    abstract_passage.put_infon('type', 'abstract')
                    abstract_passage.offset = str(title_offset)
                    abstract_passage.text = abstract_heading[2]

                # New line means end of current document
                elif line == "":
                    sorted_tags = sorted(list(term_tags), key=lambda x: x[2])
                    title_tags = filter(lambda x: x[2] <= title_offset, sorted_tags)
                    annt_tags = filter(lambda x: x[2] > title_offset, sorted_tags)
                    id_index = 0
                    # title has no annotation
                    if len(title_tags) != 0:
                        for tag in title_tags:
                            title_passage.annotations.append(get_annotations(tag, id_index))
                            id_index = id_index + 1

                    # annotation has a passage tag
                    if len(annt_tags) != 0:
                        for tag in annt_tags:
                            abstract_passage.annotations.append(get_annotations(tag, id_index))
                            id_index = id_index + 1

                    # Reset the term_Tags
                    term_tags = set([])

                    document.add_passage(title_passage)
                    document.add_passage(abstract_passage)
                    # collection.add_document(document)
                    step_parent = E('collection')
                    writer._build_documents([document], step_parent)
                    g.write(tostring(step_parent[0], pretty_print=True))
                    step_parent.clear()

                # Compile each term into a set of terms
                else:
                    terms = line.split("\t")
                    terms[1] = int(terms[1])
                    terms[2] = int(terms[2])
                    term_tags.add(tuple(terms))

        g.write(tail)

# Main
convert_pubtator("/home/davidnicholson/Documents/Data/bioconcepts2pubtator_offsets.gz", "/home/davidnicholson/Documents/Data/pubmed_docs.xml")
