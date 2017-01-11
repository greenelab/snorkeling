with open("carcinoma_pmid.txt",'r') as g:
	sents = g.readlines()[0]

ids = sents.split(',')
print len(ids)
with open("carcinoma_train_ids.txt","w") as f:
	f.write("\n".join(ids[0:4000]))

with open("Carcinoma_test_ids.txt","w") as d:
	d.write("\n".join(ids[4000:8000]))