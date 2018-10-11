import numpy as np
import math
import re
import csv
import random
import pickle


# Importing data from csv file for X and Y
trainx=[]
trainy=[]


with open('mnist/train.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			ll = [float(i) for i in row[0:-1]]
			trainx.append(ll)
			trainy.append(float(row[-1]))

# Implement Pegasos algorithm with batch size of 100

def pegasos(T,C,w,b,k,Sx,Sy):
	classes= np.unique(Sy)
	negclass = np.min(classes)
	posclass = np.max(classes)
	l = len(Sy)
	m=len(Sx[0])
	for t in range(1,T+1):
		indices=random.sample(range(0,l),k)
		At = np.take(Sx,indices,axis=0)
		yt = np.take(Sy,indices)
		for k in range(0,len(yt)):
			if yt[k]==negclass:
				yt[k]=-1
			else:
				yt[k]= 1				
		Atplus = np.zeros((0,m))
		ytplus = np.array([])
		for xa,ya in zip(At,yt):
			val = np.dot(xa,w)+b
			if (val*ya<1):
				Atplus=np.append(Atplus,[xa],axis=0)
				ytplus=np.append(ytplus,ya)
		nt=1/t
		ytplusxtplus = [ytplus[i]*Atplus[i] for i in range(0,len(ytplus))]
		ytplusxtplus = np.asarray(ytplusxtplus)
		w = np.multiply((1-nt),w)+np.multiply(((nt)*C),np.sum(ytplusxtplus,axis=0))
		b = b+((nt*C)*np.sum(ytplus))
	return np.append(w,b)	

		

l = len(trainx[0])
T = 1000
C = 1
w = np.zeros(l)
b = 0
k=100
print("Part-b")

#Extending to KC2 classifiers
classifiers = np.zeros((0,l+1))
for i in range(0,10):
	for j in range(i+1,10):
		xSet = []
		ySet = [] 
		for l,kk in zip(trainx,trainy):
			if kk==i or kk==j:
				xSet.append(l)
				ySet.append(kk)	
		xSet = np.asarray(xSet)
		ySet = np.asarray(ySet)	
		peg = pegasos(T,C,w,b,k,xSet,ySet)
		classifiers=np.append(classifiers,[peg],axis=0)





testx=[]
testy=[]
with open('mnist/test.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			ll = [float(i) for i in row[0:-1]]
			testx.append(ll)
			testy.append(float(row[-1]))

print('o2o classifier values set by learning training data')
accuracy=0
for x,y in zip(trainx,trainy):
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
	if finalprediction==y:
		accuracy+=1
	
	
train_accuracy = (accuracy/len(trainy)) *100				
print('train accuracy: ',train_accuracy,'%')




accuracy=0
for x,y in zip(testx,testy):
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
	if finalprediction==y:
		accuracy+=1
	
	
test_accuracy = (accuracy/len(testy)) *100				
print('test accuracy: ',test_accuracy,'%')




# Store by pickle

list_pickle_path = 'q21model.pkl'
list_pickle = open(list_pickle_path, 'wb')
pickle.dump(classifiers, list_pickle)
list_pickle.close()


		

