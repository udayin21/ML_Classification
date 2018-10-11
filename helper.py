import numpy as np
import math
import re
import csv
import random
import sys



testlabel=[]
predictedlabel=[]

# with open('mnist/test.csv',newline='') as csvfileX:
# 	reader = csv.reader(csvfileX)
# 	for row in reader:
# 			testlabel.append(int(row[-1]))

file = open('imdb/imdb_test_labels.txt', 'r') 
for line in file:
	testlabel.append(int(line))

file = open('imdb/q11output.txt', 'r') 
for line in file:
	predictedlabel.append(int(line))

accurate=0
count=0
for a,b in zip(testlabel,predictedlabel):
	if a==b:
		accurate+=1
	count+=1	

print(accurate/float(count))

