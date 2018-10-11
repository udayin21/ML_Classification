import numpy as np
import math
import re
import pickle

train_text = []
file = open('imdb/imdb_train_text_stemstop.txt', 'r') 
for line in file:
	# x=np.array(line.lower().split())
	x=np.array(line.replace('<br /><br />',' ').replace('(',' ').replace(')',' ').replace('-',' ').replace('.',' ').replace(',',' ').replace('/',' ').replace('\n',' ').split())
	# x=np.array(line.replace('.',' ').replace(',',' ').replace('/',' ').replace('-',' ').lower().split())
	y=x.tolist()
	train_text.append(y)


train_labels = np.array([])
file = open('imdb/imdb_train_labels.txt', 'r') 
for line in file:
	l=line.rstrip()
	train_labels=np.append(train_labels,l)


test_text = []
file = open('imdb/imdb_test_text_stemstop.txt', 'r') 
for line in file:
	# x=np.array(line.lower().split())
	x=np.array(line.replace('<br /><br />',' ').replace('(',' ').replace(')',' ').replace('-',' ').replace('.',' ').replace(',',' ').replace('/',' ').replace('\n',' ').split())
	# x=np.array(line.replace('.',' ').replace(',',' ').replace('/',' ').replace('-',' ').lower().split())
	y=x.tolist()
	test_text.append(y)


test_labels = np.array([])
file = open('imdb/imdb_test_labels.txt', 'r') 
for line in file:
	l=line.rstrip()
	test_labels=np.append(test_labels,l)	

# The classes and their Probabilities
classes=np.unique(train_labels)
pclasses={}
for i in classes:
	pclasses[i]=0

for i in train_labels:
	pclasses[i]+=1

l=len(classes)
for i in pclasses:
	pclasses[i]=float(pclasses[i])/float(len(train_labels))

# The words and their Probabilities
pwords={}
for i in pclasses:
	pwords[i]={}

count=0
vocab=0
booldic={}
for i in train_text:
	j=train_labels[count]
	classdic=pwords[j]
	for k in i:
		if k in classdic:
			classdic[k]+=1
		else:						
			classdic[k]=1	
		if not k in booldic:
			booldic[k]=1			
	count+=1	
				
nclasses={}
vocab=len(booldic)

print('vocab length='+str(vocab))



for i in classes:
	nclasses[i]=0

ll=0	
for i in train_labels:
	nclasses[i]+=len(train_text[ll])
	ll+=1

for i in nclasses:
	p=pwords[i]
	c=nclasses[i]
	for j in p:
		p[j]=(float(p[j]+1))/(float(c+vocab))


# Now check on the training data		
train_prediction=np.array([])
for i in train_text:
	argmaxprobs={}
	for j in pclasses:
		sumlog=0
		py = pclasses[j]
		c=nclasses[j]
		classdic = pwords[j]
		for k in i:
			if k in classdic:
				pxgy=classdic[k]
				sumlog+=math.log(pxgy)
			else:
				pxgy=float(1)/float(c+vocab)
				sumlog+=math.log(pxgy)	
		argmaxprobs[j]=sumlog+math.log(py)	
	train_prediction=np.append(train_prediction,max(argmaxprobs, key=argmaxprobs.get))




counter=0
accurate=0
for i in train_prediction:
	if (i==train_labels[counter]):
		accurate+=1
	counter+=1	

training_accuracy=float(accurate)/float(counter)
print('Training Accuracy: '+str(training_accuracy))

# Now check on the test data		
test_prediction=np.array([])
for i in test_text:
	argmaxprobs={}
	for j in pclasses:
		sumlog=0
		py = pclasses[j]
		c=nclasses[j]
		classdic = pwords[j]
		for k in i:
			if k in classdic:
				pxgy=classdic[k]
				sumlog+=math.log(pxgy)
			else:
				pxgy=float(1)/float(c+vocab)
				sumlog+=math.log(pxgy)	
		argmaxprobs[j]=sumlog+math.log(py)	
	test_prediction=np.append(test_prediction,max(argmaxprobs, key=argmaxprobs.get))




counter=0
accurate=0
for i in test_prediction:
	if (i==test_labels[counter]):
		accurate+=1
	counter+=1	

testing_accuracy=float(accurate)/float(counter)
print('Testing Accuracy: '+str(testing_accuracy))











