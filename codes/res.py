import numpy as np
import os
import pickle
from sklearn.metrics import accuracy_score as accuracy
import sys

folder= sys.argv[1]
files=os.listdir(folder)

acc=[]
b=[]
c=[]

file_name_dict = {}

for file in files:
	# print(file)
	epoch_no=0
	if file.endswith('.txt'):
		a = []
		b = []
		fname = file.replace('1.txt','').replace('2.txt','').replace('3.txt','').replace('4.txt','').replace('0.txt','')
		file_name_dict[fname] = {}
		file_name_dict[fname]['acc']=[]
		file_name_dict[fname]['f1']=[]

		f=open(folder+'/'+file)
		for line in f:
			line=line.strip()
			if line.startswith('macro f1'):
				line2=line.split('=')
				a.append(float(line2[1].strip()))
			if line.startswith('accuracy'):
				line2=line.split('=')
				b.append(float(line2[1].strip()))

		pos=max(a)
		epoch_no=a.index(pos)+1
		file_name_dict[fname]['f1'].append(max(a))
		file_name_dict[fname]['acc'].append(max(b))


for file in file_name_dict:
	acc = np.mean(file_name_dict[file]['acc'])
	f1  = np.mean(file_name_dict[file]['f1'])

	print(file, acc, f1)