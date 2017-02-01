from lxml import etree as ET
import tqdm
import sys

def convert(xml_file,output_file):
	""" Extract the annotations from pubtator xml formatted file
	
	Keywords arguments:
	xml_file -- The path to the xml data file
	output_file -- the path to output the formatted data
	"""
	with open(output_file, "w") as g:
		g.write("Document\tType\tID\tOffset\tEnd")

		for event, document in tqdm.tqdm(ET.iterparse(xml_file, tag="document")):
			pubmed_id = document[0].text
			
			#cycle through all the annotation tags contained within document tag
			for annotation in document.iter('annotation'):
				#not all annotations will contain an ID
				if len(annotation) > 3:
					ant_type = annotation[0]
					ant_id = annotation[3]
					offset = int(annotation[1].attrib['offset'])
					end = offset + int(annotation[1].attrib['length'])
					g.write("%s\t%s\t%s\t%s\n" % (pubmed_id, ant_type, ant_id, offset, end))
					
			#prevent memory overload
			document.clear()


#Main 
xml_file = sys.argv[1] #path to file im using now /home/davidnicholson/Documents/snorkeling/data/Epilepsy/epilepsy_data.xml
output_file = sys.argv[2] #name of file to write out
extract_annotations(xml_file,output_file)