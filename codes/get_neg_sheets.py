import pickle
import pandas as pd
import random
from tqdm import tqdm

df = pd.read_csv("../data/Negotiation - train.csv")

indices = set(df["Index"])

f = open("../data/neg_done", "r")
done = []
while True:
	line = f.readline()
	if not line: break
	done.append(line.strip())
f.close()

f = open("../data/neg.pkl", "rb")
neg_pickle = pickle.load(f)
f.close()

valid = []

for i in range(len(neg_pickle["train"])):
	valid.append(neg_pickle["train"][i]["uuid"])

for i in range(len(neg_pickle["test"])):
	valid.append(neg_pickle["test"][i]["uuid"])

for i in range(len(neg_pickle["valid"])):
	valid.append(neg_pickle["valid"][i]["uuid"])

valid = set(valid)
indices = indices - set(done)
indices = list(indices.intersection(valid))
random.shuffle(indices)

ids = {}
for i in range(5):
	start_idx = int(i * (len(indices) / 5))
	end_idx = int((i + 1) * (len(indices) / 5))
	ids[i] = indices[start_idx:end_idx]

for kf in tqdm(range(5)):
	here_ids = ids[kf]

	here = df[df['Index'].isin(here_ids)]

	here.to_csv('../data//Negotiation_'+str(kf)+'.csv')

f = open("../data/datainfo_negotiation_dict.pkl", "wb")
pickle.dump(ids, f)
f.close()