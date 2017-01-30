from lxml import etree as ET
import tqdm
import sys

xml_file = sys.argv[1]
output_file = sys.argv[2] #/home/davidnicholson/Documents/snorkeling/data/Epilepsy/epilepsy_data.xml

with open(output_file, "w") as g:
	g.write("Document\tType\tID\tOffset\tEnd")
	for event,document in tqdm.tqdm(ET.iterparse(xml_file, tag="document")):
		pubmed_id = document[0].text 
		for annotation in document.iter('annotation'):
			if len(annotation) > 3:
				ant_type = annotation[0]
				ant_id = annotation[3]
				offset = int(annotation[1].attrib['offset'])
				end = offset+int(annotation[1].attrib['length'])
				g.write("%s\t%s\t%s\t%s" %(pubmed_id,ant_type,ant_id,offset,end))
		document.clear()
