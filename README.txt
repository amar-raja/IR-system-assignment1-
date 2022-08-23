INFORMATION RETRIEVAL Assignment 1

---------------------
Table of contents
---------------------
1. Introduction
2. Prerequisites
3. Constraints
4. Approach writeup
5. Execution
6. Web links


1)Introduction
---------------
-The objective of the assignment is to retrieve relevant documents for a user
 Query efficiently by three different information retrieval systems
	a)Boolean Retreival System
	b)Tf-idf Retreival System
	c)BM25 Retreival System


2)Prerequisites
-----------------
Technical : Python,nltk,numpy,os,re,string,collections,math modules must be installed 
	    Packages to be downloaded --> - "nltk stopwords"  by --> "nltk.download('stopwords')"
				          - "nltk punkt"      by --> nltk.download('punkt')				

	    and the Linux environment to run .sh script files.
	    in LINUX BASH SHELL -- "openpyxl","xlrd","jupyter-nbconvert" are required., if not already installed.


3)Constraints
--------------

1. Extracted Corpus must be placed inside the current folder.
   - ** Files in corpus must be sorted on "Name" attribute only**.


4)Approach write-up
---------------------
      - This assignment demands 5 questions including the requirement of this "README" and "MAKE"file.
      
 Q1 ->   - Preprocessing of Corpus
		-Removed NON-ASCII characters
		-Removed Punctuations
		-Performed Tokenization     (using nltk library)
		-Removed Stopwords          (using nltk library)
		-Performed Stemming.        (using nltk library)
		

 Q2 ->   - Implemented Information Retrieval Sysytems     [irsystems.ipynb]
		c)Tf-idf retrieval system
		c)BM25 retrieval system
		c)Boolean retrieval system
 
 Q3 ->      - Created 20 queries and submitted its releveancy in QRels format.
		file = > "qrels.txt"

 Q4 ->     - Takes "40 queries txt" files as input.
	   - "40 queries txt" must be given in command line argument

	   - 40 query QRels are submitted for each ir system,
		and the resulting QRel ranking is placed into three files
			a)"boolean40.txt" b)tfidf40.txt" c)"bm2540.txt"


 Q5 ->     - README file and MAKE file ,"query.txt"

[Note-- All the input files are provided in the folder only]

MAKEFILE command= > "make run queryfile="filename.txt"

.Files Used 
-------------
1) Corpus: "english-corpora".

Files Created
-------------
1) "query.txt"      #Contains my 20 queries.

2) "qrels.txt"      #Relevancy of 20 queries in QRels format.




5)How to execute
-------------------
Individual .sh files :(file names doesn't include quotations)

	a)"twenty_queries.sh"  -->executes   "twenty_queries.ipynb"
	b)"irsystems.sh"     -->executes   "irsystems.ipynb"

  	c)"assgn1.sh"

Note: "assign1.sh" runs entire Assignment-1, it conisists of all above shell files execution.

	Execution:

	1)irsystems.ipynb contains preprocessing corpus and implementation of
	all three ir systems, this file runs around 15-20 min based on system performance.
	
	This file returns- top 10 relevant documents



6)Web Links
-------------------
Corpus --> : https://www.cse.iitk.ac.in/users/arnabb/ir/english/


