# Explanation of the repo

## Decision Tree: 

folder with all code for decision tree

#### myDT.py

My decision tree implementation following the pseudo code in lecture slides.
This is the one that was used to answer questions 2.1 to 2.6. 
Requires one argument, which is the filename of the input training set.


#### myDT_dedup.py
Mostly similar to myDT except that it contains an additional logic to prevent missing splits due to duplicate labels and arrangement in the training set.
This version is created for experimental purposes. Its output for the questions are the same as the one by myDT.py.


#### myDT_Q7.py
version of myDT that is used to answer question 2.7 and question 3.
Accepts no arguments. Use "Dbig.txt" by default.


## Lagrange: 

folder with all code for the Lagrange Interpolation.

#### call_LI.py  
run LI.py multiple times with different noise.

#### LI_datagen.py  
generate the data needed for Lagrange interpolation

#### LI.py  
performs the interpolation, draw the graph and output the errors.

#### .txt files
These are the samed test and training data to compare the output in different experiements.


## Outputs:

folder with outputs from running the code such as printed text and graphs.
