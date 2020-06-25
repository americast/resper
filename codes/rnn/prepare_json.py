from utils import *
import json
import nltk
from nltk.tokenize import word_tokenize

def save_test(outfile, filename):
	op = open(outfile,'w')

	fp = json.load(open(filename))
	data = []
	for conv_num in range(len(fp)):
		for utt_num in range(len(fp[conv_num])):

			speaker = fp[conv_num][utt_num]['speaker']

			x = normalizeString(fp[conv_num][utt_num]['utterance'])
			# x = ' '.join(word_tokenize(fp[conv_num][utt_num]['utterance'].lower()))
			if x.strip()=='':
				continue
			y = fp[conv_num][utt_num]['emotion']

			if file_type =='ER' and speaker =='1':
				continue
			elif file_type =='EE' and speaker =='0':
				continue
			# x, y, speaker = line.split("\t")
			data.append((x,y,speaker))

	for elem in data:
		op.write(elem[0]+'\t'+elem[1]+'\t'+elem[2]+'\n')

def load_data(filename, file_type):
	data = []
	cti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX} # char_to_idx
	wti = {PAD: PAD_IDX, SOS: SOS_IDX, EOS: EOS_IDX, UNK: UNK_IDX} # word_to_idx
	tti = {} # tag_to_idx
	fp = json.load(open(filename))

	for conv_num in range(len(fp)):
		for utt_num in range(len(fp[conv_num])):

			speaker = fp[conv_num][utt_num]['speaker']
			x = normalizeString(fp[conv_num][utt_num]['utterance'])

			# x = ' '.join(word_tokenize(fp[conv_num][utt_num]['utterance'].lower()))
			if x.strip()=='':
				continue
			y = fp[conv_num][utt_num]['emotion']

			if file_type =='ER' and speaker =='1':
				continue
			elif file_type =='EE' and speaker =='0':
				continue
			# x, y, speaker = line.split("\t")
			x = tokenize(x, UNIT)
			y = y.strip()
			for w in x:
				for c in w:
					if c not in cti:
						cti[c] = len(cti)
				if w not in wti:
					wti[w] = len(wti)
			if y not in tti:
				tti[y] = len(tti)
			x = ["+".join(str(cti[c]) for c in w) + ":%d" % wti[w] for w in x]
			y = [str(tti[y])]
			speaker_arr = [speaker]
			data.append(x + y+ speaker_arr)
	# fp.close()
	data.sort(key = len, reverse = True)
	return data, cti, wti, tti

if __name__ == "__main__":
	# if len(sys.argv) != 2:
	# 	sys.exit("Usage: %s training_data" % sys.argv[0])

	filename = sys.argv[2]
	folder   = sys.argv[1]

	file_type = folder.split('/')[0]

	if 'train' in filename:
		data, cti, wti, tti = load_data(filename, file_type)
		save_data(folder + ".csv", data)
		save_tkn_to_idx(folder + ".char_to_idx", cti)
		save_tkn_to_idx(folder + ".word_to_idx", wti)
		save_tkn_to_idx(folder + ".tag_to_idx", tti)
	else:
		save_test(folder+'.csv', filename)

