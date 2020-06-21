import sys; sys.path.append('common/');
from helper import *
import pudb
from random import shuffle
from tqdm import tqdm
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet

porter=PorterStemmer()
lemmatizer = WordNetLemmatizer()

def nltk2wn_tag(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:                    
    return None

def lemmatizeSentence(sentence):
  nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))    
  wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
  res_words = []
  for word, tag in wn_tagged:
    if tag is None:                        
      res_words.append(word)
    else:
      res_words.append(lemmatizer.lemmatize(word, tag))
  return " ".join(res_words)

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def scrub_words(text):
    """Basic cleaning of texts."""
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    #remove whitespace
    text=text.strip()
    return text

DATA_PATH = '../data/'

df = pd.read_csv(DATA_PATH+'/Final_annotations.csv')
df = df.rename(columns={"Our Label": "fine_labels"})

fine_labels = [i for i in list(set(df['fine_labels'])) if i==i]
fine_labels_index = {}
for i, label in enumerate(fine_labels):
  fine_labels_index[label] = i

f = open("../data/label_indices", "w")
f.write(str(fine_labels_index)+"\n")
f.close()
final_data = []

for index, row in tqdm(df.iterrows()):
  label = row['fine_labels']
  if label != label:
    continue
  # text = lemmatizeSentence(stemSentence(scrub_words(row['Unit']))).lower()
  text = row['Unit'].strip().lower()

  final_data.append([text, fine_labels_index[label]])

shuffle(final_data)

train_data = final_data[:int(0.7*(len(final_data)))]
test_data = final_data[int(0.7*(len(final_data))):]

f = open("../data/train_data_cnn", "w")
for each in train_data:
  if len(each[0]) > 0:
    f.write(each[0]+"\t"+str(each[1])+"\n")
f.close()

f = open("../data/test_data_cnn", "w")
for each in test_data:
  if len(each[0]) > 0:
    f.write(each[0]+"\t"+str(each[1])+"\n")
f.close()

f = open("../data/full_data_cnn", "w")
for each in final_data:
  if len(each[0]) > 0:
    f.write(each[0]+"\t"+str(each[1])+"\n")
f.close()

print("Complete")
# pu.db


