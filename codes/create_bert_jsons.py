from helper import *

def readUtterance(filename):
	with open(filename, encoding='utf-8') as data_file:
		data = json.loads(data_file.read())

	text_data = [[utter['utterance'] for utter in dialog] for dialog in data]
	diadata = [[normalizeString(utter['utterance']) for utter in dialog] for dialog in data]
	coarse_labels_data = [[utter['coarse_labels'] for utter in dialog] for dialog in data]
	fine_labels_data   = [[utter['fine_labels'] for utter in dialog] for dialog in data]
	resistance_labels_data= [[utter['resistance_labels'] for utter in dialog] for dialog in data]
	bertdata= [[utter['bert-feat'] for utter in dialog] for dialog in data]
	speakedata=[[utter['speaker']for utter in dialog] for dialog in data]

	vad_data=[[utter['vad_features']for utter in dialog] for dialog in data]
	affect_data=[[utter['affect_features']for utter in dialog] for dialog in data]
	emo_data=[[utter['emo_features']for utter in dialog] for dialog in data]
	liwc_data=[[utter['liwc_features']for utter in dialog] for dialog in data]
	sentiment_data=[[utter['sentiments']for utter in dialog] for dialog in data]
	fact_data=[[utter['face_acts']for utter in dialog] for dialog in data]
	# er_strategies_data=[[utter['er_strategies']for utter in dialog] for dialog in data]
	norm_er_strategies_data=[[utter['norm_er_strategies']for utter in dialog] for dialog in data]
	# er_DA_data=[[utter['er_DAs']for utter in dialog] for dialog in data]
	norm_er_DA_data=[[utter['norm_er_DAs']for utter in dialog] for dialog in data]
	ee_DA_data = [[utter['ee_DAs']for utter in dialog] for dialog in data]

	#### add the code for the BERT embeddings here and for other features

	
	return diadata, coarse_labels_data, fine_labels_data, resistance_labels_data, bertdata, speakedata, text_data, vad_data, affect_data, emo_data, liwc_data, sentiment_data, fact_data, norm_er_strategies_data, norm_er_DA_data, ee_DA_data


# Build the dict for either scripts or labels
def buildEmodict(dirt, phaselist, diadict, coarse_labels_dict, fine_labels_dict, resistance_labels_dict, count):
	""" build dicts for words and emotions """
	print("Building dicts for emotion dataset...")

	max_dialog = 0
	for phase in phaselist:
		filename = dirt + phase + str(count) +'.json'
		diadata, coarse_labels_data, fine_labels_data, resistance_labels_data, bertdata, speakedata, text_data, vad_data, affect_data, emo_data, liwc_data, sentiment_data, fact_data, norm_er_strategies_data, norm_er_DA_data, ee_DA_data  = readUtterance(filename)

		for dia, clabels, flabels, rlabels in zip(diadata, coarse_labels_data, fine_labels_data, resistance_labels_data):
			if len(dia) > max_dialog: max_dialog = len(dia)
			for d, clabel, flabel, rlabel in zip(dia, clabels, flabels, rlabels):
				diadict.addSentence(d)
				coarse_labels_dict.addWord(clabel)
				fine_labels_dict.addWord(flabel)
				resistance_labels_dict.addWord(rlabel)


	diadict.max_dialog = max_dialog

	return diadict, coarse_labels_dict, fine_labels_dict, resistance_labels_dict


# Index the tokens or the labels
def indexEmo(dirt, phase, diadict, coarse_labels_dict, fine_labels_dict, resistance_labels_dict, max_seq_len=60, count=0):

	filename = dirt + phase +str(count)+ '.json'

	diadata, coarse_labels_data, fine_labels_data, resistance_labels_data, bertdata, speakedata, text_data, vad_data, affect_data, emo_data, liwc_data, sentiment_data, fact_data, norm_er_strategies_data, norm_er_DA_data, ee_DA_data  = readUtterance(filename)

	print('Processing file {}, length {}...'.format(filename, len(diadata)))
	diaidxs = []
	coarse_label_idxs = []
	fine_label_idxs = []
	resistance_label_idxs = []

	for dia, clabels, flabels, rlabels in zip(diadata, coarse_labels_data, fine_labels_data, resistance_labels_data):
	# for dia, emo, bert, speaker, donor in zip(diadata, emodata, bertdata, speakerdata, donordata):
		dia_idxs = []
		emo_idxs = []
		cl_idxs  = []
		fl_idxs  = []
		rl_idxs  = []

		# bert_feats=[]
		for d, cl, fl, rl in zip(dia, clabels, flabels, rlabels):
		
			d_idxs = [diadict.word2index[w] if w in diadict.word2index else UNK for w in d.split(' ')]  # MELD and EmoryNLP not used for building vocab
			# e_idxs = [emodict.word2index[e]]
			cl_idx = [coarse_labels_dict.word2index[cl]]
			fl_idx = [fine_labels_dict.word2index[fl]]
			rl_idx = [resistance_labels_dict.word2index[rl]]


			if len(d_idxs) > max_seq_len:
				dia_idxs.append(d_idxs[:max_seq_len])
			else:
				dia_idxs.append(d_idxs + [PAD] * (max_seq_len - len(d_idxs)))
			# emo_idxs.append(e_idxs)
			cl_idxs.append(cl_idx)
			fl_idxs.append(fl_idx)
			rl_idxs.append(rl_idx)

		diaidxs.append(dia_idxs)
		# emoidxs.append(emo_idxs)
		coarse_label_idxs.append(cl_idxs)
		fine_label_idxs.append(fl_idxs)
		resistance_label_idxs.append(rl_idxs)

	diafield = dict()

	diafield['feat'] = diaidxs
	diafield['coarse_labels'] =  coarse_label_idxs
	diafield['fine_labels']   =  fine_label_idxs
	diafield['resistance_labels'] =  resistance_label_idxs
	diafield['bert-feat']= bertdata
	diafield['speaker']=speakedata
	diafield['text'] = text_data
	diafield['vad_features']= vad_data
	diafield['affect_features']=affect_data
	diafield['emo_features']  = emo_data
	diafield['liwc_features'] = liwc_data
	diafield['sentiment_features']= sentiment_data
	diafield['face_features'] = fact_data
	diafield['norm_er_strategies']= norm_er_strategies_data
	diafield['norm_er_DAs']= norm_er_DA_data
	diafield['ee_DAs']= ee_DA_data
	
	return diafield


# Overall preprocessing function
def proc_emoset(dirt, phaselist, emoset, min_count, max_seq_len, count):
	""" Build data from emotion sets """

	diadict = Dictionary('dialogue')
	coarse_labels_dict = Dictionary('coarse_labels')
	fine_labels_dict = Dictionary('fine_labels')
	resistance_labels_dict = Dictionary('resistance_labels')

	diadict, coarse_labels_dict, fine_labels_dict, resistance_labels_dict = buildEmodict(dirt=dirt, phaselist=phaselist, diadict=diadict, coarse_labels_dict= coarse_labels_dict, fine_labels_dict=fine_labels_dict, resistance_labels_dict=resistance_labels_dict, count=count)

	diadict.delRare(min_count=min_count, padunk=True)

	coarse_labels_dict.delRare(min_count=0, padunk=False)
	fine_labels_dict.delRare(min_count=0, padunk=False)
	resistance_labels_dict.delRare(min_count=0, padunk=False)

	dump_pickle(dirt+emoset+str(count)+ '_vocab_bert.pt', diadict)

	print('Dialogue vocabulary (min_count={}): majority words {} rare words {}\n'.format(
		min_count, diadict.n_words, len(diadict.rare)))

	dump_pickle(dirt+emoset +str(count) + '_coarse_labels_dict_bert.pt', coarse_labels_dict)
	dump_pickle(dirt+emoset +str(count) + '_fine_labels_dict_bert.pt', fine_labels_dict)
	dump_pickle(dirt+emoset +str(count) + '_resistance_labels_dict_bert.pt', resistance_labels_dict)

	# print('Emotions:\n {}\n {}\n'.format(emodict.word2index, emodict.word2count))

	# add the emodict for training set
	tr_diadict = Dictionary('dialogue_tr')
	tr_coarse_labels_dict= Dictionary('coarse_labels_tr')
	tr_fine_labels_dict= Dictionary('fine_labels_tr')
	tr_resistance_labels_dict= Dictionary('resistance_labels_tr')

	# tr_emodict = Dictionary('emotion_tr')
	tr_diadict, tr_coarse_labels_dict, tr_fine_labels_dict, tr_resistance_labels_dict = buildEmodict(dirt=dirt, phaselist=['train'], diadict=tr_diadict, coarse_labels_dict= tr_coarse_labels_dict, fine_labels_dict=tr_fine_labels_dict, resistance_labels_dict= tr_resistance_labels_dict, count=count)


	tr_diadict.delRare(min_count=min_count, padunk=True)
	tr_coarse_labels_dict.delRare(min_count=0, padunk=False)
	tr_fine_labels_dict.delRare(min_count=0, padunk=False)
	tr_resistance_labels_dict.delRare(min_count=0, padunk=False)

	
	dump_pickle(dirt+emoset +str(count) + '_tr_coarse_labels_dict_bert.pt', tr_coarse_labels_dict)
	dump_pickle(dirt+emoset +str(count) + '_tr_fine_labels_dict_bert.pt', tr_fine_labels_dict)
	dump_pickle(dirt+emoset +str(count) + '_tr_resistance_labels_dict_bert.pt', tr_resistance_labels_dict)	


	# index and put into fields
	Datafield = dict()
	for phase in phaselist:
		diafield = indexEmo(dirt=dirt, phase=phase, diadict=diadict, coarse_labels_dict = coarse_labels_dict, fine_labels_dict= fine_labels_dict, resistance_labels_dict=resistance_labels_dict, max_seq_len=max_seq_len, count=count)
		Datafield[phase] = diafield

	data_path = emoset + '_data.pt'
	
	dump_pickle(dirt+emoset+str(count)+'_bert_data.pt' , Datafield)
	
	return 1


def main():
	''' Main function '''
	parser = argparse.ArgumentParser()
	parser.add_argument('-min_count', type=int, default = 0)
	parser.add_argument('-max_seq_len', type=int, default=80)
	parser.add_argument('-count', type= int, default=0)
	emoset = 'resisting'

	opt = parser.parse_args()

	print(opt, '\n')

	phaselist = ['train', 'test']

	dirt = '../data/higru_bert_data/'
	proc_emoset(dirt=dirt, phaselist=phaselist, emoset=emoset, min_count=opt.min_count, max_seq_len=opt.max_seq_len, count= opt.count)


if __name__ == '__main__':

	main()