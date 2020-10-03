import os
import numpy as np
import sys
import csv
import pudb

dir_name = '../../data/higru_bert_data/results/'

files = os.listdir(dir_name)

dataset = sys.argv[1]

file_dict = {}

for file in sorted(files):
	# print(file)
	fp = open(dir_name + file)

	if dataset == 'neg':
		file = file.replace('negotiation0','').replace('negotiation1','').replace('negotiation2','').replace('negotiation3','').replace('negotiation4','')
	elif dataset == 'res_old':
		file = file.replace('resisting_old0','').replace('resisting_old1','').replace('resisting_old2','').replace('resisting_old3','').replace('resisting_old4','')
	else:	
		file = file.replace('resisting0','').replace('resisting1','').replace('resisting2','').replace('resisting3','').replace('resisting4','')
		
	if file not in file_dict:
		file_dict[file]={}
		file_dict[file]['acc']=[]
		file_dict[file]['don_acc']=[]
		file_dict[file]['f1']=[]
		file_dict[file]['don_f1']=[]

	accs=[]
	don_accs=[]
	f1s=[]
	don_f1s=[]
	don = False
	# if "don" in file:
	# 	pu.db
	for line in fp:
		line = line.strip().split('\t')
		try:
			accs.append(float(line[1]))
			f1s.append(float(line[2]))
			don_accs.append(float(line[3]))
			don_f1s.append(float(line[4]))
			don = True
		except Exception as e:
			try:
				accs.append(float(line[1]))
				f1s.append(float(line[2]))
			except:
				continue

	if not accs or not f1s:
		continue

	# file_dict[file]['acc'].append(max(accs))
	max_f_pos= f1s.index(max(f1s))
	if don: max_don_f_pos= don_f1s.index(max(don_f1s))

	file_dict[file]['f1'].append(f1s[max_f_pos])
	file_dict[file]['acc'].append(accs[max_f_pos])
	if don:
		file_dict[file]['don_f1'].append(don_f1s[max_f_pos])
		file_dict[file]['don_acc'].append(don_accs[max_f_pos])
	else:
		file_dict[file]['don_f1'].append(-1)
		file_dict[file]['don_acc'].append(-1)


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
	elif 'combo' in file:
		model_type = 'combo'
	
	return model_type
	

EE_seed_file_dict={}
ER_seed_file_dict={}
all_seed_file_dict={}


output_list = []
for file in sorted(file_dict):
	if len(file_dict[file]['f1']) != 5:
		continue

	model_type = get_model_type(file)	
	acc = round(np.mean(file_dict[file]['acc']),2)
	f1  = round(np.mean(file_dict[file]['f1']),2)
	sd  = round(np.std(file_dict[file]['f1']),2)
	don_acc = round(np.mean(file_dict[file]['don_acc']),2)
	don_f1 = round(np.mean(file_dict[file]['don_f1']),2)

	record = {"file": file, "acc": round(np.mean(file_dict[file]['acc']),3), "f1": round(np.mean(file_dict[file]['f1']),3)}
	output_list.append(record)


	output_file_name = "file_acc_f1_neg.csv" if dataset == "neg" else "file_acc_f1_res.csv"
	print('{}\t{}\t{}\t{}'.format(file, round(np.mean(file_dict[file]['acc']),3), round(np.mean(file_dict[file]['f1']),3), round(np.std(file_dict[file]['f1']),3)),round(np.mean(file_dict[file]['don_acc']),3), round(np.mean(file_dict[file]['don_f1']),3))

with open(output_file_name, 'w', encoding='utf8', newline='') as output_file:
    csv_w = csv.DictWriter(output_file, fieldnames=output_list[0].keys(),)
    csv_w.writeheader()
    csv_w.writerows(output_list)

# for model in sorted(all_seed_file_dict):
# 	ER_acc = ER_seed_file_dict[model][0]
# 	ER_f1  = ER_seed_file_dict[model][1]
# 	EE_acc = EE_seed_file_dict[model][0]
# 	EE_f1  = EE_seed_file_dict[model][1]
# 	all_acc = all_seed_file_dict[model][0]
# 	all_f1  = all_seed_file_dict[model][1]
	
# 	print(model+'&'+ str(ER_acc)+'&'+str(ER_f1)+'&'+str(EE_acc)+'&'+str(EE_f1)+'&'+str(all_acc)+'&'+str(all_f1)+'\\\\')



## train the task level models. Last hidden utterance state of the softmax-classifier ==> donation probability. 