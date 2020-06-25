import pandas as pd
import numpy as np
import os

infolder='../ER_final_Conv_data/'
outfolder='ER_data/'


for i in range(0,5):
	df= pd.read_csv(infolder+'train'+str(i)+'.csv')	
	train_text_file=open(outfolder+'train'+str(i)+'.txt','w')
	for id_,row in df.iterrows():
		unit= row['Unit'].lower().strip().replace('\t',' ')
		label= row['one_label']
		train_text_file.write(unit+'\t'+str(label)+'\n')

	df= pd.read_csv(infolder+'test'+str(i)+'.csv')	
	train_text_file=open(outfolder+'test'+str(i)+'.txt','w')
	for id_,row in df.iterrows():
		unit= row['Unit'].lower().strip().replace('\t',' ')
		label= row['one_label']
		train_text_file.write(unit+'\t'+str(label)+'\n')
	

