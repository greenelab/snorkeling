import csv
from lxml import etree as ET
import sys


import tqdm


def extract_annotations(xml_file, output_file):
    """ Extract the annotations from pubtator xml formatted file

    Keywords arguments:
    xml_file -- The path to the xml data file
    output_file -- the path to output the formatted data
    """
    with open(output_file, "w") as csvfile:
        fieldnames = ['Document', 'Type', 'ID', 'Offset', 'End']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for event, document in tqdm.tqdm(ET.iterparse(xml_file, tag="document")):
            pubmed_id = document[0].text

            # cycle through all the annotation tags contained within document tag
            for annotation in document.iter('annotation'):
                # not all annotations will contain an ID
                if len(annotation) > 3:
                    for infon in annotation.iter('infon'):
                        if infon.attrib["key"] == "type":
                            ant_type = infon.text
                        else:
                            ant_id = infon.text

                    for location in annotation.iter('location'):
                        offset = int(location.attrib['offset'])
                        end = offset + int(location.attrib['length'])

                    writer.writerow({'Document': pubmed_id, 'Type': ant_type, 'ID': ant_id, 'Offset': offset, 'End': end})

            # prevent memory overload
            document.clear()


# Main
xml_file = sys.argv[1]  # path to file im using now /home/davidnicholson/Documents/snorkeling/data/Epilepsy/epilepsy_data.xml
output_file = sys.argv[2]  # name of file to write out
extract_annotations(xml_file, output_file)
