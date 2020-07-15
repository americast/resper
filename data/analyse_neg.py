import pandas as pd
import pudb
from nltk.tokenize import word_tokenize
from tqdm import tqdm

df = pd.read_csv("Negotiation - Final Annotation - Final annotations.csv")


convs = list(set(df["Index"])) 

uttrs = []
num_tokens = []
words = []
for each in tqdm(convs):
	train = df[df['Index'].isin([each])]
	num_uttrs = len(train)
	uttrs.append(num_uttrs)
	for i, row in train.iterrows():
		text = train.loc[i]["Text"]
		tokens_here = word_tokenize(text)
		num_tokens.append(len(tokens_here))
		words.extend(tokens_here)


pu.db