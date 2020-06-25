import sys
import re
from time import time
from os.path import isfile
from parameters import *
from collections import defaultdict

def normalize(x):
    # x = re.sub("[\uAC00-\uD7A3]+", "\uAC00", x) £ convert Hangeul to 가
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        return x
    if unit == "word":
        return x.split(" ")

def save_data(filename, data):
    fo = open(filename, "w")
    for seq in data:
        fo.write(" ".join(seq) + "\n")
    fo.close()

def load_tkn_to_idx(filename):
    print("loading %s" % filename)
    tkn_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        tkn_to_idx[line] = len(tkn_to_idx)
    fo.close()
    return tkn_to_idx

def load_idx_to_tkn(filename):
    print("loading %s" % filename)
    idx_to_tkn = []
    fo = open(filename)
    for line in fo:
        line = line[:-1]
        idx_to_tkn.append(line)
    fo.close()
    return idx_to_tkn

def save_tkn_to_idx(filename, tkn_to_idx):
    fo = open(filename, "w")
    for tkn, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tkn)
    fo.close()

def load_checkpoint(filename, model = None):
    print("loading %s" % filename)
    checkpoint = torch.load(filename)
    if model:
        model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving %s" % filename)
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def cudify(f):
    return lambda *x: f(*x).cuda() if CUDA else f(*x)

Tensor = cudify(torch.Tensor)
LongTensor = cudify(torch.LongTensor)
zeros = cudify(torch.zeros)

def maskset(x):
    mask = x.eq(PAD_IDX)
    return (mask, x.size(1) - mask.sum(1)) # set of mask and lengths

def idx_to_tkn(tkn_to_idx):
    return [x for x, _ in sorted(tkn_to_idx.items(), key = lambda x: x[1])]

def batchify(xc, xw, sos = False, eos = False, minlen = 0):
    xw_len = max(minlen, max(len(x) for x in xw))
    if xc:
        xc_len = max(minlen, max(len(w) for x in xc for w in x))
        pad = [[PAD_IDX] * (xc_len + 2)]
        xc = [[[SOS_IDX] + w + [EOS_IDX] + [PAD_IDX] * (xc_len - len(w)) for w in x] for x in xc]
        xc = [(pad if sos else []) + x + (pad * (xw_len - len(x) + eos)) for x in xc]
        xc = LongTensor(xc)
    sos = [SOS_IDX] if sos else []
    eos = [EOS_IDX] if eos else []
    xw = [sos + list(x) + eos + [PAD_IDX] * (xw_len - len(x)) for x in xw]
    return xc, LongTensor(xw)

def heatmap(m, x, itw, ch = True, rh = False, sos = False, eos = False): # attention heatmap
    f = "%%.%df" % NUM_DIGITS
    m = [v[:len(x) + sos + eos] for v in m] # remove padding
    m = [([SOS] if sos else []) + [itw[i] for i in x] + ([EOS] if eos else [])] + m
    if ch: # column header
        csv = DELIM.join([x for x in m[0]]) + "\n" # source sequence
    for row in m[ch:]:
        if rh: # row header
            csv += row[0] + DELIM # target sequence
        csv += DELIM.join([f % x for x in row[rh:]]) + "\n"
    return csv

def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0
