import numpy as np
import math
import re
import random
import pickle
import sys

# Pickle to load the model
list_pickle_path = 'q13model.pkl'
list_unpickle = open(list_pickle_path, 'rb')
classes = pickle.load(list_unpickle)
vocab = pickle.load(list_unpickle)
nclasses = pickle.load(list_unpickle)
pclasses = pickle.load(list_unpickle)
pwords = pickle.load(list_unpickle)
frequentwords = pickle.load(list_unpickle)
list_unpickle.close()


inputfilename = sys.argv[1]
outputfilename = sys.argv[2]
test_text = []
file = open(inputfilename, 'r') 
for line in file:
	x=np.array(line.lower().replace('<br /><br />',' ').replace('(',' ').replace(')',' ').replace('-',' ').replace('.',' ').replace(',',' ').replace('/',' ').replace('\n',' ').split())
	y=x.tolist()
	test_text.append(y)


# Now check on the test data		
test_prediction=np.array([])
for i in test_text:
	argmaxprobs={}
	for j in pclasses:
		sumlog=0
		py = pclasses[j]
		c=nclasses[j]
		classdic = pwords[j]
		bigrams = [' '.join(w) for w in zip(i[:-1],i[1:])]
		for k in bigrams:
		# for k in i:
			if k in classdic and k not in frequentwords:
				pxgy=classdic[k]
				sumlog+=math.log(pxgy)
			else:
				pxgy=float(1)/float(c+vocab)
				sumlog+=math.log(pxgy)	
		argmaxprobs[j]=sumlog+math.log(py)	
	test_prediction=np.append(test_prediction,max(argmaxprobs, key=argmaxprobs.get))


file = open(outputfilename,'w') 
for p in test_prediction:
	file.write(str(p)+'\n')
file.close()	




	
