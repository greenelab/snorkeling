from Bio import Entrez
Entrez.email = "test@aol.com"
#handle = Entrez.esearch(db="pubmed",term="hox")
#record = Entrez.read(handle)
#print record
handle = Entrez.efetch(db="pubmed", id=10021330,retmode="xml")
with open("test.xml","w") as f:
	f.write(handle.read())
#records = Entrez.read(handle)
#print records

