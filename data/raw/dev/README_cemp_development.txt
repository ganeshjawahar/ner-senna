CHEMDNER-patents CEMP subtask development text data corrected version (Version 10th July 2015)
------------------------------------------------------------------------

For additional questions please send e-mail to: Martin Krallinger (mkrallinger@cnio.es)
or to the Biocreative participant mailing list: http://biocreative.sourceforge.net/mailing.html

This directory contains the development set text for the CHEMDNER-patents CEMP subtask.

1) chemdner_patents_development_text.txt : training set

This file contains plain-text, UTF8-encoded Patent abstracts in a 
tab-separated format with the following three columns:

1- Patent identifier
2- Title of the patent
3- Abstract of the patent

In total 7000 abstracts are provided in this development set (7000 titles and 7000 abstracts)

2) chemdner_cemp_gold_standard_development_v02.tsv

For the CEMP (chemical entity mention in patents) task we distribute manually tagged patents (title and abstracts) with 
structure-associated chemical entity mentions (SACEMs). The CEMP annotations consist of tab-separated fields containing:

1- Patent identifier
2- Type of text from which the annotation was derived (T: Title, A: Abstract)
3- Start offset
4- End offset
5- Text string of the entity mention
6- Type of chemical entity mention (ABBREVIATION,FAMILY,FORMULA,IDENTIFIERS,MULTIPLE,SYSTEMATIC,TRIVIAL)


3) chemdner_cemp_gold_standard_development_eval_v02.tsv

Gold standard evaluation format to be used for assessment with the biocreative evaluation script.

It consists of tab-separated columns containing:

1- Patent identifier
2- Offset string consisting in a triplet joined by the ':' character. You have to provide the text type (T: Title, A:Abstract), the start offset and the end offset.


4) CEMP task prediction format

For the CEMP task we will only request the prediction of the chemical mention offsets following
a similar stetting as done for the BioCreative IV CHEMDNER task on PubMed abstracts. Given a set
of patent abstracts, the participants have to return the start and end indices corresponding to 
all the chemical entities mentioned in this document. 

It consists of tab-separated columns containing:

1- Patent identifier
2- Offset string consisting in a triplet joined by the ':' character. You have to provide the text type (T: Title, A:Abstract), the start offset and the end offset.
3- The rank of the chemical entity returned for this document
4- A confidence score
5- The string of the chemical entity mention

An example illustrating the prediction format is shown below:

WO2009026621A1	A:12:24	1	0.99	paliperidone
WO2011115938A1	T:0:17	1	0.99	Spiro-tetracyclic
WO2011115687A2	A:0:12	1	0.99	SP-B
WO2011115687A2	T:0:22	2	0.98989	Alkylated
WO2011115687A2	A:104:117	3	0.98978	SP-B
US20050101595	A:0:13	1	0.99	Aminothiazole
US20050101595	A:60:67	2	0.98989	2-amino
US20050101595	T:0:50	3	0.98978	N-containing
US20050101595	A:29:52	4	0.98967	N-containing
WO2010147138A1	A:252:262	1	0.99	nucleotide
WO2010147138A1	A:363:373	2	0.98989	amino
WO2010147138A1	A:92:102	3	0.98978	fatty
CN103087254A	A:196:218	1	0.99	stearyl


The evaluation will be done using the BioCreative Evaluation script available at:

http://www.biocreative.org/resources/biocreative-ii5/evaluation-library/

In this case the INT - article classification format option will be used.

Example command:
 bc-evaluate --INT team_cemp_prediction.tsv chemdner_cemp_gold_standard_development_eval.tsv > team_cemp_prediction.eval


where --INT corresponds to the required evaluation option
team_cemp_prediction.tsv 				corresponds to the prediction file (in the correct format specified above)
chemdner_cemp_gold_standard_development_eval.tsv	corresponds to the evaluation file


If you have problems with the required prediction format use bc-evaluate with the flag --debug to find out what is wrong.












