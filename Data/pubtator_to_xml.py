import gzip

import tqdm


def generate_annotation(file_object,tagged_term,index):
	"""Generates the annotation tag

	Keyword Arguments:
	file_object -- the file object used for writing 
	tagged_term -- look up 
	index -- keep track of the annotation id
	"""
	file_object.write("<annotation id=\"%d\">\n" % (index))
	file_object.write("<infon key=\"title\">%s</infon>\n" %(tagged_term[4]))
	file_object.write("<location length=\"%d\" offset=\"%d\"></location>\n" % (len(tagged_term[3]),tagged_term[1]))
	file_object.write("<text>%s</text>\n" % (tagged_term[3]))

	# If the annotation type is a Gene,Species, Mutation, SNP
	# Write out relevant tag
	if tagged_term[4] == "Gene":
		file_object.write("<infon key=\"NCBI Gene\">%s</infon>\n" % (tagged_term[5][0:tagged_term[5].find("(")]))

	elif tagged_term[4] == "Species":
		file_object.write("<infon key=\"NCBI Species\">%s</infon>\n" % (tagged_term[5]))

	elif "Mutation" in tagged_term[4]:
		file_object.write("<infon key=\"tmVar\">%s</infon>\n" % (tagged_term[5]))

	elif "SNP" in tagged_term[4]:
		file_object.write("<infon key=\"tmVar\">%s</infon>\n" %(tagged_term[5]))

	else:
		# If there is no MESH ID for an annotation	
		if len(tagged_term) > 5:
			# check to see if there are multiple mesh tags
			if "|" in tagged_term[5]:
				# Write out each MESH id as own tag
				for ids in tagged_term[5].split("|"):
					# Some ids dont have the MESH:#### form so added case to that
					if ":" not in ids:
						file_object.write("<infon key=\"MESH\">%s</infon>\n" % (tagged_term[5]))
					else:
						term_type,term_id = ids.split(":")
						file_object.write("<infon key=\"%s\">%s</infon>\n" % (term_type,term_id))
			else:
				# Some ids dont have the MESH:#### form so added case to that
				if ":" in tagged_term[5]:
					term_type,term_id = tagged_term[5].split(":")
					file_object.write("<infon key=\"%s\">%s</infon>\n" % (term_type,term_id))
				else:
					file_object.write("<infon key=\"MESH\">%s</infon>\n" % (tagged_term[5]))
		else:
			file_object.write("<infon key=\"MESH\">Unknown</infon>\n")
	file_object.write("</annotation>\n")

def convert_pubtator(input_file,output_file):
	"""Convert pubtators annotation list to XML

	Keyword Arguments:
	input_file -- the path of pubtators annotation file
	output_file -- the path to output the converted text
	"""
	term_tags = set()
	with open(output_file,'w') as g:
		# Header for pubtator's xml format
		g.write("<collection>\n<source>Pubtator</source>\n")
		# read from gunzip file
		with gzip.open(input_file,"rb") as f:

			for line in tqdm.tqdm(f):
				#Convert "illegal chracters" (i.e. < > &) in the main text into html entities
				line = line.strip().replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

				# Title parsing
				if "|t|" in line:
					title_heading = line.split("|")
					has_written_title = False
				
				# Abstract parsing
				elif "|a|" in line:
					abstract_heading = line.split("|")
					title_offset = len(title_heading[2])
					has_written_abstract = False

				# New line means end of current document
				elif line == "":
					sorted_tags = sorted(list(term_tags),key=lambda x: x[2])

					#Fixing the issue where titles don't have an annotation
					if len(term_tags) == 0 or sorted_tags[0][2] > title_offset:
						if not(has_written_title):
							g.write("<document>\n")
							g.write("<id>" + title_heading[0] + "</id>\n")
							g.write("<passage>\n")
							g.write("<infon key=\"type\">title</infon>\n")
							g.write("<offset>0</offset>\n")
							g.write("<text>" + title_heading[2] + "</text>\n")
							has_written_title = True

					# For every annotation
					for i,tagged_term in enumerate(sorted_tags):

						# Determine if annotation is for the title else abstract
						if tagged_term[2] <= title_offset:
							if not(has_written_title):
								g.write("<document>\n")
								g.write("<id>" + title_heading[0] + "</id>\n")
								g.write("<passage>\n")
								g.write("<infon key=\"type\">title</infon>\n")
								g.write("<offset>0</offset>\n")
								g.write("<text>" + title_heading[2] + "</text>\n")
								has_written_title = True
							generate_annotation(g,tagged_term,i)
						# Abastract
						else:
							if not(has_written_abstract):
								g.write("</passage>\n<passage>\n")
								g.write("<infon key=\"type\">abstract</infon>\n")
								g.write("<offset>%d</offset>\n" %(title_offset))
								g.write("<text>" + abstract_heading[2] + "</text>\n")
								has_written_abstract = True
							generate_annotation(g,tagged_term,i)

					# Fixes the issue where articles don't have an abstract
					if not(has_written_abstract):
						g.write("</passage>\n<passage>\n")
						g.write("<infon key=\"type\">abstract</infon>\n")
						g.write("<offset>%d</offset>\n" %(title_offset))
						g.write("<text>" + abstract_heading[2] + "</text>\n")
						has_written_abstract = True

					# Close the passage block
					g.write("</passage>\n</document>\n")

					# Reset the term_Tags
					term_tags = set([])

				# Compile each term into a set of terms
				else:
					terms = line.split("\t")
					terms[1] = int(terms[1])
					terms[2] = int(terms[2])
					term_tags.add(terms)
		# Close the whole collection
		g.write("</collection>\n")

convert_pubtator("bioconcepts2pubtator_offsets.gz","pubmed_docs.xml")