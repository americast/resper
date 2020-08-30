""" Main function """
import os
import argparse
import Utils
import Const
import sys; sys.path.append('../')
from helper import *
# from Preprocess import Dictionary # import the object for pickle loading
from Modules import *
from EmoTrain import emotrain, emoeval, emotrain_combo, emoeval_combo
from datetime import datetime
import math
import time
import random
import pdb


def seed_everything(seed=100):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def main():
	'''Main function'''

	parser = argparse.ArgumentParser()

	# Learning
	parser.add_argument('-lr', type=float, default=2.5e-4)		# Learning rate: 2.5e-4 for Friends and EmotionPush, 1e-4 for IEMOCAP
	parser.add_argument('-decay', type=float, default=math.pow(0.5, 1/40))	# half lr every 20 epochs
	parser.add_argument('-epochs', type=int, default=200)		# Defualt epochs 200
	parser.add_argument('-patience', type=int, default=10,		# Patience of early stopping 10 epochs
						help='patience for early stopping')
	parser.add_argument('-save_dir', type=str, default="../../data/higru_bert_data/models",	# Save the model and results in snapshot/
						help='where to save the models')
	# Data
	parser.add_argument('-dataset', type=str, default='Teaching0',	
						help='dataset')
	parser.add_argument('-data_path', type=str, required = True,
					   help='data path')
	parser.add_argument('-vocab_path', type=str, required=True,
						help='vocabulary path')
	parser.add_argument('-emodict_path', type=str, required=True,
						help='emotion label dict path')
	parser.add_argument('-tr_emodict_path', type=str, default=None,
						help='training set emodict path')
	parser.add_argument('-max_seq_len', type=int, default=80,	# Pad each utterance to 80 tokens
						help='the sequence length')
	# model
	parser.add_argument('-label_type', type=str, default='coarse',
						help='particular type pf labels used i.e coarse/fine/resistance')

	parser.add_argument('-type', type=str, default='higru', 	# Model type: default HiGRU 
						help='choose the low encoder')
	parser.add_argument('-d_word_vec', type=int, default=300,	# Embeddings size 300
						help='the word embeddings size')
	parser.add_argument('-d_h1', type=int, default=300,		# Lower-level RNN hidden state size 300
						help='the hidden size of rnn1')
	parser.add_argument('-d_h2', type=int, default=300,		# Upper-level RNN hidden state size 300
						help='the hidden size of rnn1')
	parser.add_argument('-d_fc', type=int, default=100,		# FC size 100
						help='the size of fc')
	parser.add_argument('-gpu', type=str, default=None,		# Spcify the GPU for training
						help='gpu: default 0')
	parser.add_argument('-embedding', type=str, default=None,	# Stored embedding path
						help='filename of embedding pickle')
	parser.add_argument('-report_loss', type=int, default=720,	# Report loss interval, default the number of dialogues
						help='how many steps to report loss')
	parser.add_argument('-bert', type=int, default=0,	# Report loss interval, default the number of dialogues
						help='include bert or not')

	# parser.add_argument('-mask', type=str, default='all',	# Choice of mask for ER, EE, or all
	# 					help='include mask type')


	
	# parser.add_argument('-alpha', type=float, default=0.9,	# proportion of the loss , 0.9 means the loss for the Face acts
	# 					help='include mask type')
	
	# parser.add_argument('-interpret', type=str, default='single_loss', # combined trainable loss, 
	# 					help ='name of the file to be saved')

	# parser.add_argument('-ldm', type=int, default=1, help = 'how many last utterances used for the donor loss contribution') # last donor mask
	# parser.add_argument('-don_model', type=int, default=1, help = 'how to compute the donation probability') # last donor mask

	# parser.add_argument('-thresh_reg', type=float, default=0.0, help = 'how to choose threshold for the models 2 and 3') # last donor mask

	parser.add_argument('-bert_train', type=int, default=0, help = 'choose 0 or 1') # last donor mask

	parser.add_argument('-addn_features', type =str, default='all', help='include all possible features') # include the features to be used for training

	parser.add_argument('-seed', type =int, default=100, help= 'set random seed')

	args = parser.parse_args()
	print(args, '\n')

	seed_everything(args.seed)
	if 'resisting' in args.dataset:
		# feature_dim_dict = {'vad_features': 3, 'affect_features': 4, 'emo_features': 10, 'liwc_features': 64, 'sentiment_features': 3, 'face_features': 8, 'norm_er_strategies': 10, 'norm_er_DAs': 17, 'ee_DAs': 23, 'all': 3+4+10+64+3+8+10+17+23}
		feature_dim_dict = {'vad_features': 3, 'affect_features': 4, 'emo_features': 10, 'liwc_features': 64, 'sentiment_features': 3, 'all': 3+4+10+64+3}

	elif 'negotiation' in args.dataset:
		feature_dim_dict = {'vad_features': 3, 'affect_features': 4, 'emo_features': 10, 'liwc_features': 64, 'sentiment_features': 3, 'all': 3+4+10+64+3}

	feature_dim = 0
	if args.addn_features in feature_dim_dict:
		feature_dim = feature_dim_dict[args.addn_features]


	# Load vocabs
	print("Loading vocabulary...")
	worddict = Utils.loadFrPickle(args.vocab_path)
	print("Loading emotion label dict...")
	emodict = Utils.loadFrPickle(args.emodict_path)
	print("Loading review tr_emodict...")
	tr_emodict = Utils.loadFrPickle(args.tr_emodict_path)

	# Load data field
	print("Loading field...")
	field = Utils.loadFrPickle(args.data_path)
	
	test_loader = field['test']

	# import pdb; pdb.set_trace()
	trainable=False
	if args.bert_train==1:
		trainable =True

	# Initialize word embeddings
	print("Initializing word embeddings...")
	embedding = nn.Embedding(worddict.n_words, args.d_word_vec, padding_idx=Const.PAD)
	if args.d_word_vec == 300:
		if args.embedding != None and os.path.isfile(args.embedding):
			print("Loading previous saved embeddings")
			np_embedding = Utils.loadFrPickle(args.embedding)
		else:
			np_embedding = Utils.load_pretrain(args.d_word_vec, worddict, type='glove')
			Utils.saveToPickle(args.embedding, np_embedding)
		embedding.weight.data.copy_(torch.from_numpy(np_embedding))
	embedding.weight.requires_grad = trainable
	# pu.db
	# Choose the model
	if args.type.startswith('combo'):
		print("Training the combo model")
		model_bin = combo_bin(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type+"_bin",
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)
		model_multi = combo_multi(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type+"_multi",
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)
	elif args.type.startswith('bert-lstm'):
		print("Word-level BiGRU baseline")
		model = BERT_LSTM(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)
	elif args.type.startswith('bert-higru-sent-attn-mask'):
		print("Training sentence-based masked attention")
		model = BERT_HiGRU_sent_attn_mask(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)
	elif args.type.startswith('bert-higru-sent-conn-mask'):
		print("Training sentence-based masked connections")
		model = BERT_HiGRU_sent_conn_mask(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	# Choose the model
	elif args.type.startswith('bert-higru-sent-attn-2'):
		print("Training sentence-based attention second")
		model = BERT_HiGRU_sent_attn_2(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	elif args.type.startswith('bert-higru-sent-attn'):
		print("Training sentence-based attention")
		model = BERT_HiGRU_sent_attn(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	elif args.type.startswith('bert-higru-uttr-attn-2'):
		print("Training utterance-based attention double level")
		model = BERT_HiGRU_uttr_attn_2(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	elif args.type.startswith('bert-higru-uttr-attn-3'):
		print("Training utterance-based attention second level only")
		model = BERT_HiGRU_uttr_attn_3(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	elif args.type.startswith('bert-higru-uttr-attn'):
		print("Training utterance-based attention")
		model = BERT_HiGRU_uttr_attn(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	elif args.type.startswith('bert-higru'):
		model = BERT_HiGRU(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:],
					  # bert_flag= args.bert,
					  # don_model= args.don_model,
					  trainable= trainable,
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )
					  #speaker_flag= args.sf)

	elif args.type.startswith('higru'):
		model = HiGRU(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type,
					  # bert= args.bert,
					  # don_model= args.don_model
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )


	elif args.type.startswith('bert-bigru'):
		model = BERT_BiGRU(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type[5:].replace('bigru','higru'),
					  # bert_flag= args.bert,
					  # don_model= args.don_model
					  trainable=trainable,
					  feature_dim = feature_dim
,
long_bert = args.bert
					  )


	elif args.type.startswith('bigru'):
		model = BiGRU(d_word_vec=args.d_word_vec,
					  d_h1=args.d_h1,
					  d_h2=args.d_h2,
					  d_fc=args.d_fc,
					  emodict=emodict,
					  worddict=worddict,
					  embedding=embedding,
					  type=args.type.replace('bigru','higru'),
					  # bert= args.bert,
					  # don_model= args.don_model
					  feature_dim = feature_dim,
					  long_bert = args.bert
					  )


	# elif args.type.startswith('bert-gru'):
	# 	model = BERT_BiGRU(d_word_vec=args.d_word_vec,
	# 				  d_h1=args.d_h1,
	# 				  d_h2=args.d_h2,
	# 				  d_fc=args.d_fc,
	# 				  emodict=emodict,
	# 				  worddict=worddict,
	# 				  embedding=embedding,
	# 				  type=args.type[5:].replace('gru','higru'),
	# 				  bert_flag= args.bert,
	# 				  don_model= args.don_model)


	# Choose focused emotions

	focus_emo = []

	# Train the model
	if args.type.startswith('combo'):
		emotrain_combo(model_bin=model_bin,
			 model_multi=model_multi,
			 data_loader=field,
			 tr_emodict=tr_emodict,
			 emodict=emodict,
			 args=args,
			 focus_emo=focus_emo)
	else:
		emotrain(model=model,
			 data_loader=field,
			 tr_emodict=tr_emodict,
			 emodict=emodict,
			 args=args,
			 focus_emo=focus_emo)

	# Load the best model to test
	print("Load best models for testing!")

	file_str = Utils.return_file_path(args)
	# model = model.load_state_dict(args.save_dir+'/'+file_str+'.pt', map_location='cpu')
	if args.type.startswith('combo'):
		model_bin   = torch.load(args.save_dir+'/'+file_str+'_model_bin.pt', map_location='cpu')
		model_multi = torch.load(args.save_dir+'/'+file_str+'_model_multi.pt', map_location='cpu')
		pAccs, acc, mf1, = emoeval_combo(model_bin=model_bin,
						model_multi=model_multi,
						data_loader=test_loader,
						tr_emodict=tr_emodict,
						emodict=emodict,
						args=args,
						focus_emo=focus_emo)
	else:
		model = torch.load(args.save_dir+'/'+file_str+'_model.pt', map_location='cpu')
		# model = torch.load_state_dict(args.save_dir+'/'+file_str+'.pt', map_location='cpu')
		pAccs, acc, mf1, = emoeval(model=model,
						data_loader=test_loader,
						tr_emodict=tr_emodict,
						emodict=emodict,
						args=args,
						focus_emo=focus_emo)


	print("Test: ACCs-WA-UWA {}".format(pAccs))
	print("Accuracy = {}, F1 = {}".format(acc, mf1))

	# Save the test results
	record_file = '{}/{}_{}.txt'.format(args.save_dir, args.type, args.label_type)
	if os.path.isfile(record_file):
		f_rec = open(record_file, "a")
	else:
		f_rec = open(record_file, "w")

	f_rec.write("{} - {} - {}\t:\t{}\n".format(datetime.now(), args.d_h1, args.lr, pAccs))
	f_rec.close()


if __name__ == '__main__':
	main()
