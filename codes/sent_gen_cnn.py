import sys; sys.path.append('common/');
from helper import *
import pudb
from random import shuffle
from tqdm import tqdm


DATA_PATH = '../data/'

df = pd.read_csv(DATA_PATH+'/Final_annotations.csv')
df = df.rename(columns={"Our Label": "fine_labels"})

fine_labels = [i for i in list(set(df['fine_labels'])) if i==i]

final_data = []

for index, row in tqdm(df.iterrows()):
	text = row['Unit']
	label = row['fine_labels']
	if label != label:
		continue
	final_data.append([text, fine_labels.index(label)])

shuffle(final_data)

train_data = final_data[:int(0.7*(len(final_data)))]
test_data = final_data[int(0.7*(len(final_data))):]

f = open("../data/train_data_cnn", "w")
for each in train_data:
	f.write(each[0]+"\t"+str(each[1])+"\n")
f.close()

f = open("../data/test_data_cnn", "w")
for each in test_data:
	f.write(each[0]+"\t"+str(each[1])+"\n")
f.close()

f = open("../data/full_data_cnn", "w")
for each in final_data:
	f.write(each[0]+"\t"+str(each[1])+"\n")
f.close()

print("Complete")
# pu.db


