import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import sys

train_set_label='../cemp_training_set/chemdner_cemp_gold_standard_train.tsv'
train_set_text='../cemp_training_set/chemdner_patents_train_text.txt'
dev_set_label='../cemp_development_set_v03/chemdner_cemp_gold_standard_development_v03.tsv'
dev_set_text='../cemp_development_set_v03/chemdner_patents_development_text.txt'
test_set_text='/home/ganesh/Desktop/TIR/ravindra/test.set'

def preprocess1(label_file,text_file,dest_file):
	#Read the label file
	label_map={}
	with open(label_file,"r") as ins:
		array=[]
		for line in ins:
			content=line.strip().split('\t')
			if content[0] not in label_map:
				label_map[content[0]]=[]
			label_map[content[0]].append(content[1]+':'+content[4])
	print(len(label_map))
	#Read the text file
	title_map={}
	text_map={}
	id_map={}
	with open(text_file,"r") as ins:
		array=[]
		row=0
		for line in ins:
			content=line.strip().split('\t')
			if content[0] not in title_map:
				title_map[content[0]]=content[1]
			if content[0] not in text_map:
				text_map[content[0]]=content[2]
			id_map[row]=content[0]
			row=row+1
	#Tokenize the corpus
	res_file=open(dest_file,'w')
	err=0
	aec=0
	tec=0
	for i in range(len(id_map)):
		p_id=id_map[i]
		title=title_map[p_id]
		text=text_map[p_id]
		labels=[]
		if p_id in label_map:
			labels=label_map[p_id]
		line=p_id+'\n'
		try:
			#abstract
			out,laec=embed_labels_a_line(title,'A',labels)
			line+=out+'\n'
			aec+=laec
			#text
			out,ltec=embed_labels_mul_line(text,'T',labels)
			line+=out+'\n'
			tec+=ltec
			res_file.write(line)
		except UnicodeDecodeError:
			err=err+1
		except Exception:
			print(sys.exc_info()[0])
			sys.exit(0)
	res_file.close()

	print('#lines with unicode errors: '+str(err))
	print('#total abstract entites found: '+str(aec))
	print('#total text entites found: '+str(tec))

def get_mentions(full_text,typ,labels):
	chem_mentions=[]
	if len(labels)>0:
		for label in labels:
			if label.startswith(typ):
				content=label.split(':')
				chem_mentions.append(content[1])
	return chem_mentions

def get_word_label(target_word,chem_mentions):
	if len(chem_mentions)==0:
		return target_word+'$$$O'
	for mention in chem_mentions:
		if target_word.startswith(mention):
			return target_word+'$$$I'
	return target_word+'$$$O'

def embed_labels_a_line(full_text,typ,labels):
	words=word_tokenize(full_text.encode('utf-8'))
	chem_mentions=get_mentions(full_text,typ,labels)
	res=''
	in_count=0
	for word in words:
		new_word=get_word_label(word,chem_mentions)
		if new_word.endswith('I'):
			in_count=in_count+1
		res=res+new_word+' '
	return res.strip(),in_count

def embed_labels_mul_line(sentences,typ,labels):
	sent_tokenize_list=sent_tokenize(sentences)
	res=str(len(sent_tokenize_list))+'\n'
	in_count=0
	for sentence in sent_tokenize_list:
		out,inc=embed_labels_a_line(sentence,typ,labels)
		in_count+=inc
		res+=out+'\n'
	return res.strip(),in_count

def get_words(sentence):
	words=word_tokenize(sentence.encode('utf-8'))
	res=''
	for word in words:
		res+=word+' '
	return res.strip()

def get_sentences(sentence):
	res=''
	sent_tokenize_list=sent_tokenize(sentence)
	for s in sent_tokenize_list:
		res+=get_words(s)+'\n'
	return res.strip(),len(sent_tokenize_list)

def preprocess_2(text_file,dest_file):
	res_file=open(dest_file,'w')
	err=0
	with open(text_file,"r") as ins:
		array=[]
		for line in ins:
			try:
				content=line.strip().split('\t')
				out=content[0]+'\n'+get_words(content[1])+'\n'
				text,count=get_sentences(content[2])
				out+=str(count)+'\n'+text+'\n'
				res_file.write(out)
			except UnicodeDecodeError:
				err=err+1
	res_file.close()
	print('#Lines with unicode errros: '+str(err))


#preprocess1(train_set_label,train_set_text,'../data/train.tsv')
#preprocess1(dev_set_label,dev_set_text,'../data/dev.tsv')
preprocess_2(test_set_text,'../data/test.tsv')

text="this is a sentence. i am ganesh."
sent_tokenize_list=sent_tokenize(text)
print(sent_tokenize_list)
print(word_tokenize('Hello World.'))