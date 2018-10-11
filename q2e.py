import numpy as np
import math
import re
import csv
import random
import sys



testlabel=[]
predictedlabel=[]

with open('mnist/test.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			testlabel.append(int(row[-1]))

file = open('A2SampleInputOutputFiles/q23output.txt', 'r') 
for line in file:
	predictedlabel.append(int(line))


accurate=0
count=0
for a,b in zip(testlabel,predictedlabel):
	if a==b:
		accurate+=1
	count+=1	


print('Confusion matrix')
counter=0
accurate=0
confusion_matrix = np.zeros((10,10))

for a,b in zip(testlabel,predictedlabel):
	confusion_matrix[b][a]+=1

for i in range(0,10):
	print(confusion_matrix[i])	

