from lxml import etree as ET
import sys
import os
import gzip

import tqdm


root = ET.Element("collection")
tree = ET.ElementTree(root)
source = ET.SubElement(root,"source")
source.text = "Pubtator"
term_tags = set()

def generate_annotation(file_object,tagged_term,index):
	"""generates the annotation tag

	Keyword Arguments:
	file_object -- the file object used for writing 
	tagged_term -- look up 
	index -- keep track of the annotation id
	"""
	file_object.write("<annotation id=\"%d\">\n" % (index))
	file_object.write("<infon key=\"title\">%s</infon>\n" %(tagged_term[4]))
	file_object.write("<location length=\"%d\" offset=\"%d\"></location>\n" % (len(tagged_term[3]),tagged_term[1]))
	file_object.write("<text>%s</text>\n" % (tagged_term[3]))

	if tagged_term[4] == "Gene":
		file_object.write("<infon key=\"NCBI Gene\">%s</infon>\n" % (tagged_term[5][0:tagged_term[5].find("(")]))

	elif tagged_term[4] == "Species":
		file_object.write("<infon key=\"NCBI Species\">%s</infon>\n" % (tagged_term[5]))

	elif "Mutation" in tagged_term[4]:
		file_object.write("<infon key=\"tmVar\">%s</infon>\n" % (tagged_term[5]))

	elif "SNP" in tagged_term[4]:
		file_object.write("<infon key=\"tmVar\">%s</infon>\n" %(tagged_term[5]))
	else:
		if len(tagged_term) > 5:
			if "|" in tagged_term[5]:
				for ids in tagged_term[5].split("|"):
					if ":" not in ids:
						file_object.write("<infon key=\"MESH\">%s</infon>\n" % (tagged_term[5]))
					else:
						term_type,term_id = ids.split(":")
						file_object.write("<infon key=\"%s\">%s</infon>\n" % (term_type,term_id))
						#ET.SubElement(word_annotation,"infon",{"key":term_type}).text = term_id
			else:
				if ":" in tagged_term[5]:
					term_type,term_id = tagged_term[5].split(":")
					file_object.write("<infon key=\"%s\">%s</infon>\n" % (term_type,term_id))
					#ET.SubElement(word_annotation,"infon",{"key":term_type}).text = term_id
				else:
					file_object.write("<infon key=\"MESH\">%s</infon>\n" % (tagged_term[5]))
					#ET.SubElement(word_annotation,"infon",{"key":"MESH"}).text = tagged_term[5]
		else:
			file_object.write("<infon key=\"MESH\">Unknown</infon>\n")
			#term_type = "MESH"
			#ET.SubElement(word_annotation,"infon",{"key":term_type}).text="Unknown"
	file_object.write("</annotation>\n")

docs = 0
with open('text.xml','w') as g:
	g.write("<collection>\n<source>Pubtator</source>\n")
	with gzip.open("bioconcepts2pubtator_offsets.gz","rb") as f:
		for line in tqdm.tqdm(f):
			line = line.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')
			if "|t|" in line:
				title_heading = line.split("|")
				has_written_title = False
			elif "|a|" in line:
				abstract_heading = line.split("|")
				title_offset = len(title_heading[2])
				has_written_abstract = False
			elif line == "":
				sorted_tags = sorted(list(term_tags),key=lambda x: x[2])
				
				#Hack for fixing the issue where titles don't have an annotation
				if len(term_tags) == 0 or sorted_tags[0][2] > title_offset:
					if not(has_written_title):
						g.write("<document>\n")
						g.write("<id>" + title_heading[0] + "</id>\n")
						g.write("<passage>\n")
						g.write("<infon key=\"type\">title</infon>\n")
						g.write("<offset>0</offset>\n")
						g.write("<text>" + title_heading[2] + "</text>\n")
						has_written_title = True

				for i,tagged_term in enumerate(sorted_tags):
					if tagged_term[2] <= title_offset:
						if not(has_written_title):
							g.write("<document>\n")
							g.write("<id>" + title_heading[0] + "</id>\n")
							g.write("<passage>\n")
							g.write("<infon key=\"type\">title</infon>\n")
							g.write("<offset>0</offset>\n")
							g.write("<text>" + title_heading[2] + "</text>\n")
							has_written_title = True
							docs = docs + 1
						generate_annotation(g,tagged_term,i)
					else:
						if not(has_written_abstract):
							g.write("</passage>\n<passage>\n")
							g.write("<infon key=\"type\">abstract</infon>\n")
							g.write("<offset>%d</offset>\n" %(title_offset))
							g.write("<text>" + abstract_heading[2] + "</text>\n")
							has_written_abstract = True
						generate_annotation(g,tagged_term,i)

				#hack for fixing the issue where articles don't have an abstract
				if not(has_written_abstract):
					g.write("</passage>\n<passage>\n")
					g.write("<infon key=\"type\">abstract</infon>\n")
					g.write("<offset>%d</offset>\n" %(title_offset))
					g.write("<text>" + abstract_heading[2] + "</text>\n")
					has_written_abstract = True

				g.write("</passage>\n</document>\n")
				#reset the term_Tags
				term_tags = set([])
			else:
				terms = line.split("\t")
				terms[1] = int(terms[1])
				terms[2] = int(terms[2])
				term_tags.add(tuple(terms))
	g.write("</collection>\n")
