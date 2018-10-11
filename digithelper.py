import numpy as np
import math
import re
import csv
import random
import sys



trainx=[]
trainy=[]

filename=sys.argv[1]

with open(filename,newline='') as csvfileX:
	reader = csv.reader(csvfileX)
	for row in reader:
			ll = [int(i) for i in row]
			trainx.append(ll)
			trainy.append(int(0))


print('converting to libsvm:')
f = open('libsvm_form.txt','w')
for row,val in zip(trainx,trainy):
	s=str(val)+' '
	index=1
	for r in row:
		s = s+str(index)+':'+str(r)+' '
		index = index + 1
	f.write(s+'\n')
f.close()			

