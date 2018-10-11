import numpy as np
import math
import re
import csv
import random



trainx=[]
trainy=[]

with open('mnist/train.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			ll = [int(i) for i in row[0:-1]]
			trainx.append(ll)
			trainy.append(int(row[-1]))


testx=[]
testy=[]
with open('mnist/test.csv',newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			ll = [int(i) for i in row[0:-1]]
			testx.append(ll)
			testy.append(int(row[-1]))

print('writing now:')
f = open('train_libsvm.txt','w')
for row,val in zip(trainx,trainy):
	s=str(val)+' '
	index=1
	for r in row:
		s = s+str(index)+':'+str(r)+' '
		index = index + 1
	f.write(s+'\n')
f.close()
	

print('writing now:')
f = open('test_libsvm.txt','w')
for row,val in zip(testx,testy):
	s=str(val)+' '
	index=1
	for r in row:
		s = s+str(index)+':'+str(r)+' '
		index = index + 1
	f.write(s+'\n')
f.close()	
