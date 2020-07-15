import pandas as pd
import pudb
from nltk.tokenize import word_tokenize
from tqdm import tqdm

df = pd.read_csv("Final_annotations.csv")

convs = list(set(df["B2"])) 

uttrs = []
num_tokens = []
words = []
for each in tqdm(convs):
	train = df[df['B2'].isin([each])]
	num_uttrs = len(train)
	uttrs.append(num_uttrs)
	for i, row in train.iterrows():
		text = train.loc[i]["Unit"]
		tokens_here = word_tokenize(text)
		num_tokens.append(len(tokens_here))
		words.extend(tokens_here)


pu.db