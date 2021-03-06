import os
import numpy as np
import sys

dir_name='../../data/higru_bert_data/results_seqmodel/'

files= os.listdir(dir_name)

file_dict={}

for file in sorted(files):
	# print(file)
	fp= open(dir_name+file)

	file = file.replace('count_0','').replace('count_1','').replace('count_2','').replace('count_3','').replace('count_4','')
		
	if file not in file_dict:
		file_dict[file]={}
		file_dict[file]['acc']=[]
		file_dict[file]['f1']=[]

	

	accs=[]
	f1s=[]
	

	for line in fp:
		line= line.strip().split('\t')
		try:
			accs.append(float(line[1]))
			f1s.append(float(line[2]))
		except Exception as e:
			continue

	if accs==[] or f1s==[]:
		continue

	# file_dict[file]['acc'].append(max(accs))
	max_f_pos = 0
	max_f_pos= f1s.index(max(f1s))

	file_dict[file]['f1'].append(f1s[max_f_pos])
	file_dict[file]['acc'].append(accs[max_f_pos])
	# file_dict[file]['don_f1'].append(don_f1s[max_f_pos])
	# file_dict[file]['don_acc'].append(don_accs[max_f_pos])


def get_model_type(file):
	model_type = None

	if 'bert-higru-sf' in file:
		model_type = 'bert-higru-sf'
	elif 'bert-higru-f' in file:
		model_type = 'bert-higru-f'
	elif 'bert-higru' in file:
		model_type = 'bert-higru'
	elif 'bert-bigru-sf' in file:
		model_type = 'bert-bigru-sf'
	elif 'bert-bigru-f' in file:
		model_type = 'bert-bigru-f'
	elif 'bert-bigru' in file:
		model_type = 'bert-bigru'
	elif 'higru-sf' in file:
		model_type = 'higru-sf'
	elif 'higru-f' in file:
		model_type = 'higru-f'
	elif 'higru' in file:
		model_type = 'higru'
	elif 'bigru-sf' in file:
		model_type = 'bigru-sf'
	elif 'bigru-f' in file:
		model_type = 'bigru-f'
	elif 'bigru' in file:
		model_type = 'bigru'
	
	return model_type
	

EE_seed_file_dict={}
ER_seed_file_dict={}
all_seed_file_dict={}

for file in sorted(file_dict):
	if len(file_dict[file]['f1'])!=5:
		continue

	model_type = get_model_type(file)	
	acc = round(np.mean(file_dict[file]['acc']),2)
	f1  = round(np.mean(file_dict[file]['f1']),2)

	# if 'EE' in file:
	# 	EE_seed_file_dict[model_type] = (acc, f1)
	# elif 'ER' in file:
	# 	ER_seed_file_dict[model_type] = (acc, f1)
	# elif 'all' in file:
		# all_seed_file_dict[model_type] = (acc, f1)


	print('{}\t{}\t{}'.format(file, round(np.mean(file_dict[file]['acc']),3), round(np.mean(file_dict[file]['f1']),3)))#,round(np.mean(file_dict[file]['don_acc']),3), round(np.mean(file_dict[file]['don_f1']),3)))


# for model in sorted(all_seed_file_dict):
# 	ER_acc = ER_seed_file_dict[model][0]
# 	ER_f1  = ER_seed_file_dict[model][1]
# 	EE_acc = EE_seed_file_dict[model][0]
# 	EE_f1  = EE_seed_file_dict[model][1]
# 	all_acc = all_seed_file_dict[model][0]
# 	all_f1  = all_seed_file_dict[model][1]
	
# 	print(model+'&'+ str(ER_acc)+'&'+str(ER_f1)+'&'+str(EE_acc)+'&'+str(EE_f1)+'&'+str(all_acc)+'&'+str(all_f1)+'\\\\')



## train the task level models. Last hidden utterance state of the softmax-classifier ==> donation probability. 