import os, sys, pdb, numpy as np, pandas as pd, pickle, time, codecs, requests, argparse, random, math, scipy, sklearn, scipy.stats
from glob import glob
from bs4 import BeautifulSoup
from requests import get
import io, zipfile, tarfile

from joblib import Parallel, delayed

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

def load_pickle(file):
	with open(file,'rb') as handle:
		return pickle.load(handle)

def dump_pickle(file, obj):
	with open(file,'wb') as handle:
		pickle.dump(file, handle)

DATA_PATH = '/projects/persuasionforgood-master/Face_acts/dialogue_act_prediction/resisting-persuasion/data'


from collections import defaultdict as ddict
