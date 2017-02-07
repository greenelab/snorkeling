import argparse
import csv
import sys

from lxml import etree as ET
import tqdm


def extract_annotations(xml_file, output_file):
    """ Extract the annotations from pubtator xml formatted file
    Outputs a TSV file with the following header terms:
    Document - the corresponding pubmed id
    Type - the type of term (i.e. Chemical, Disease, Gene etc.)
    ID - the appropiate MESH or NCBI ID if known
    Offset - the character position where the term starts
    End - the character position where the term ends

    Keywords arguments:
    xml_file -- The path to the xml data file
    output_file -- the path to output the formatted data
    """
    with open(output_file, "w") as csvfile:
        fieldnames = ['Document', 'Type', 'ID', 'Offset', 'End']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        tag_generator = ET.iterparse(xml_file, tag="document")

        for event, document in tqdm.tqdm(tag_generator):
            pubmed_id = document[0].text

            # cycle through all the annotation tags contained within document tag
            for annotation in document.iter('annotation'):

                # not all annotations will contain an ID
                if len(annotation) <= 3:
                    continue

                for infon in annotation.iter('infon'):
                    if infon.attrib["key"] == "type":
                        ant_type = infon.text
                    else:
                        ant_id = infon.text

                location, = annotation.iter('location')
                offset = int(location.attrib['offset'])
                end = offset + int(location.attrib['length'])
                row = {'Document': pubmed_id, 'Type': ant_type, 'ID': ant_id, 'Offset': offset, 'End': end}
                writer.writerow(row)

            # prevent memory overload
            document.clear()


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts the annotations from the BioC xml format')
    parser.add_argument("--input", help="File path pointing to input file.", type=str)
    parser.add_argument("--output", nargs="?", help="File path for destination of output.")

    args = parser.parse_args()

    # check if user gave arguments
    if not(args.input):
        raise Exception("PLEASE GIVE FILE INPUT PATH")

    if not(args.output):
        raise Exception("PLEASE GIVE FILE OUTPUT PATH")

    extract_annotations(args.input, args.output)
