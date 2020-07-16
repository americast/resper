import os, sys, pdb, numpy as np, pandas as pd, pickle, time, codecs, requests, argparse, random, math, scipy, sklearn, scipy.stats
from glob import glob
from bs4 import BeautifulSoup
from requests import get
import io, zipfile, tarfile
from collections import defaultdict as ddict
from joblib import Parallel, delayed
import csv, unicodedata, re,json


def mergeList(list_of_1):
	a= []
	for elem in list_of_1:
		a.append(elem)
	return a

def get_chunks_size(inp_list, chunk_size):
	return [inp_list[x:x+chunk_size] for x in range(0, len(inp_list), chunk_size)]

def get_chunks_num(inp_list, num_chunks):

	chunk_size = int(len(inp_list)/num_chunks)
	return [inp_list[int(x* chunk_size):min(int((x+1)*chunk_size),len(inp_list))] for x in range(num_chunks)]

DATA_PATH = '/projects/persuasionforgood-master/Face_acts/dialogue_act_prediction/resisting-persuasion/data'


def load_pickle(file):
	with open(file,'rb') as handle:
		return pickle.load(handle)

def dump_pickle(file, obj):
	with open(file,'wb') as handle:
		pickle.dump(obj, handle)

PAD = 0
UNK = 1
PAD_WORD = '<pad>'
UNK_WORD = '<unk>'


# Definition of the dictionary class to be used.
class Dictionary:
	def __init__(self, name):
		self.name = name
		self.pre_word2count = {}
		self.rare = []
		self.word2count = {}
		self.word2index = {}
		self.index2word = {}
		self.n_words = 0
		self.max_length = 0
		self.max_dialog = 0

	# delete the rare words by the threshold min_count
	def delRare(self, min_count, padunk=True):

		# collect rare words
		for w,c in self.pre_word2count.items():
			if c < min_count:
				self.rare.append(w)

		# add pad and unk
		if padunk:
			self.word2index[PAD_WORD] = PAD
			self.index2word[PAD] = PAD_WORD
			self.word2count[PAD_WORD] = 1
			self.word2index[UNK_WORD] = UNK
			self.index2word[UNK] = UNK_WORD
			self.word2count[UNK_WORD] = 1
			self.n_words += 2

		# index words
		for w,c in self.pre_word2count.items():
			if w not in self.rare:
				self.word2count[w] = c
				self.word2index[w] = self.n_words
				self.index2word[self.n_words] = w
				self.n_words += 1

	def addSentence(self, sentence):
		sentsplit = sentence.split(' ')
		if len(sentsplit) > self.max_length:
			self.max_length = len(sentsplit)
		for word in sentsplit:
			self.addWord(word)

	def addWord(self, word):
		if word not in self.pre_word2count:
			self.pre_word2count[word] = 1
		else:
			self.pre_word2count[word] += 1


# Normalize strings
def unicodeToAscii(str):
	return ''.join(
		c for c in unicodedata.normalize('NFD', str)
		if unicodedata.category(c) != 'Mn'
	)


# Remove nonalphabetics
def normalizeString(str):
	str = unicodeToAscii(str.lower().strip())
	str = re.sub(r"([!?])", r" \1", str)
	str = re.sub(r"[^a-zA-Z0-9!?]+", r" ", str)
	return str


# from sentence_splitter import SentenceSplitter, split_text_into_sentences
# splitter = SentenceSplitter(language='en')

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import json

from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel
from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
