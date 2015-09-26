import nltk
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import sys

train_set_label='../data/raw/train/chemdner_cemp_gold_standard_train.tsv'
train_set_text='../data/raw/train/chemdner_patents_train_text.txt'
dev_set_label='../data/raw/dev/chemdner_cemp_gold_standard_development_v03.tsv'
dev_set_text='../data/raw/dev/chemdner_patents_development_text.txt'
test_set_text='../data/raw/test/test.set'
new_train_file='../data/preprocessed/train.tsv'
new_dev_file='../data/preprocessed/dev.tsv'
new_test_file='../data/preprocessed/test.tsv'

def preprocess1(label_file,text_file,dest_file):
	#Read the label file
	label_map={}
	title_ents=0
	abs_ents=0	
	with open(label_file,"r") as ins:
		array=[]
		for line in ins:
			content=line.strip().split('\t')
			if content[0] not in label_map:
				label_map[content[0]]=[]
			label_map[content[0]].append(content[1]+':'+content[4])
			if content[1]=='T':
				title_ents=title_ents+1
			else:
				abs_ents=abs_ents+1
	print('#chemical entities : '+str(title_ents+abs_ents))
	#Read the text file
	title_map={}
	abs_map={}
	id_map={}
	with open(text_file,"r") as ins:
		array=[]
		row=0
		for line in ins:
			content=line.strip().split('\t')
			if content[0] not in title_map:
				title_map[content[0]]=content[1]
			if content[0] not in abs_map:
				abs_map[content[0]]=content[2]
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
		text=abs_map[p_id]
		labels=[]
		if p_id in label_map:
			labels=label_map[p_id]
		line=p_id+'\n'
		try:
			#title
			out,ltec=embed_labels_a_line(title,'T',labels)
			line+=out+'\n'
			tec+=ltec
			#abstract
			out,laec=embed_labels_mul_line(text,'A',labels)
			line+=out+'\n'
			aec+=laec
			res_file.write(line)
		except UnicodeDecodeError:
			err=err+1
		except Exception:
			print(sys.exc_info()[0])
			sys.exit(0)
	res_file.close()

	print('#patents: '+str(len(abs_map)))
	print('#patents with atleast one entity mention: '+str(len(label_map)))
	print('#patents with unicode errors: '+str(err))
	print('#title unigram entities found: '+str(tec)+' out of '+str(title_ents)+' ngram entities')
	print('#abstract unigram entities found: '+str(aec)+' out of '+str(abs_ents)+' ngram entities')

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
		if target_word in mention:
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
	print('#Lines with unicode errors: '+str(err))

print('---TRAIN STATISTICS---')
preprocess1(train_set_label,train_set_text,new_train_file)
print('\n---DEV STATISTICS---')
preprocess1(dev_set_label,dev_set_text,new_dev_file)
print('\n---TEST STATISTICS---')
preprocess_2(test_set_text,new_test_file)