import numpy as np
import math
import re
import random
import pickle
import sys
import csv

# Pickle to load the model
list_pickle_path = 'q21model.pkl'
list_unpickle = open(list_pickle_path, 'rb')
classifiers = pickle.load(list_unpickle)
list_unpickle.close()

inputfilename = sys.argv[1]
outputfilename = sys.argv[2]

testx=[]
with open(inputfilename,newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			ll = [float(i) for i in row]
			testx.append(ll)

file = open(outputfilename,'w') 
accuracy=0
for x in testx:
	count=0
	prediction=np.zeros(10)
	for i in range(0,10):
		for j in range(i+1,10):
			getclassifier=classifiers[count]
			val = np.dot(getclassifier[0:-1],x)+(getclassifier[-1])
			if val<0:
				prediction[i]+=1
			else:
				prediction[j]+=1	
			count+=1				
	finalpredictionval = max(prediction)
	finalprediction=0
	for k in range(0,10):
		if prediction[k]==finalpredictionval:
			finalprediction=k		
	file.write(str(finalprediction)+'\n')
	
file.close()			