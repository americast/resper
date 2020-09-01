
""" functions, layers, and architecture of HiGRU """
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import Const


# Dot-product attention
def get_attention(q, k, v, attn_mask=None):
	"""
	:param : (batch, seq_len, seq_len)
	:return: (batch, seq_len, seq_len)
	"""
	attn = torch.matmul(q, k.transpose(1, 2))
	if attn_mask is not None:
		attn.data.masked_fill_(attn_mask, -1e10)

	attn = F.softmax(attn, dim=-1)
	output = torch.matmul(attn, v)
	return output, attn

def get_sent_attention(q,k,v, sent_attn_mask):

	attn = torch.matmul(q,k.transpose(0,1))
	attn = attn.mul(sent_attn_mask)
	attn = F.softmax(attn, dim=-1)
	output = torch.matmul(attn,v)

	return output, attn


# Get mask for attention
def get_attn_pad_mask(seq_q, seq_k):
	assert seq_q.dim() == 2 and seq_k.dim() == 2

	pad_attn_mask = torch.matmul(seq_q.unsqueeze(2).float(), seq_k.unsqueeze(1).float())
	pad_attn_mask = pad_attn_mask.eq(Const.PAD)  # b_size x 1 x len_k
	#print(pad_attn_mask)

	return pad_attn_mask.cuda(seq_k.device)

def get_word_pad_attns(seq_q):

	pad_word_mask = seq_q.clone()
	pad_word_mask[pad_word_mask!=0]=1

	return pad_word_mask.cuda(seq_q.device)

	# pad_word_mask = np.ones((seq_q.shape[0],seq_q.shape[1]))

	# for i in range(seq_q.shape[0]):
	# 	for j in range(seq_q.shape[1]):
	# 		if seq_q[i,j] ==0:
	# 			pad_word_mask[i,j]=0

	# pad_word_mask = torch.FloatTensor(pad_word_mask).cuda(seq_q.device)

	# return pad_word_mask

def get_sent_pad_attn(seq_q):
	pad_sent_mask = np.zeros((seq_q.shape[0], seq_q.shape[0]))
	for i in range(seq_q.shape[0]):
		pad_sent_mask[i,:(i+1)] =     1
		pad_sent_mask[i,i+1:]   = -1e10

	pad_sent_mask = torch.FloatTensor(pad_sent_mask).cuda(seq_q.device)
	return pad_sent_mask



# Pad for utterances with variable lengths and maintain the order of them after GRU
class GRUencoder(nn.Module):
	def __init__(self, d_emb, d_out, num_layers):
		super(GRUencoder, self).__init__()
		# default encoder 2 layers
		self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
						  bidirectional=True, num_layers=num_layers, dropout=0.3)

	def forward(self, sent, sent_lens):
		"""
		:param sent: torch tensor, batch_size x seq_len x d_rnn_in
		:param sent_lens: numpy tensor, batch_size x 1
		:return:
		"""
		device = sent.device
		# seq_len x batch_size x d_rnn_in
		sent_embs = sent.transpose(0,1)

		# sort by length
		s_lens, idx_sort = np.sort(sent_lens)[::-1], np.argsort(-sent_lens)
		s_lens =s_lens.copy()
		idx_unsort = np.argsort(idx_sort)


		idx_sort = torch.from_numpy(idx_sort).cuda(device)
		s_embs = sent_embs.index_select(1, Variable(idx_sort))

		# padding
		sent_packed = pack_padded_sequence(s_embs, s_lens)
		sent_output = self.gru(sent_packed)[0]
		sent_output = pad_packed_sequence(sent_output, total_length=sent.size(1))[0]

		# unsort by length
		idx_unsort = torch.from_numpy(idx_unsort).cuda(device)
		sent_output = sent_output.index_select(1, Variable(idx_unsort))

		# batch x seq_len x 2*d_out
		output = sent_output.transpose(0,1)

		return output


# Utterance encoder with three types: higru, higru-f, and higru-sf
class UttEncoder(nn.Module):
	def __init__(self, d_word_vec, d_h1, type):
		super(UttEncoder, self).__init__()
		self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1)
		self.d_input = 2 * d_h1
		self.model = type
		if '-f' not in self.model:
			self.d_input = 4 * d_h1 + d_word_vec
		else:
			self.d_input = 2 * d_h1 + d_word_vec
		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h1),
			nn.Tanh()
		)

	def forward(self, sents, lengths, sa_mask=None):
		"""
		:param sents: batch x seq_len x 2*d_h1
		:param lengths: numpy array 1 x batch
		:return: batch x d_h1
		"""
		w_context = self.encoder(sents, lengths)
		combined = w_context

		if '-f' not in self.model:
			w_lcont, w_rcont = w_context.chunk(2, -1)
			sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
			sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
			combined = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
			combined = torch.cat(combined, dim=-1)
		else:
			w_lcont, w_rcont = w_context.chunk(2, -1)
			combined = [w_lcont, sents, w_rcont]
			combined = torch.cat(combined, dim=-1)


		output1 = self.output1(combined)
		output = torch.max(output1, dim=1)[0]

		return output

# Utterance encoder with three types: higru, higru-f, and higru-sf
class UttEncoder_attn(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, type):
		super(UttEncoder_attn, self).__init__()
		self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1)
		self.d_h1, self.d_h2 = d_h1, d_h2
		self.d_input = 2 * d_h1
		self.model = type
		if self.model == 'higru-f':
			self.d_input = 2 * d_h1 + d_word_vec
		if self.model == 'higru-sf' or self.model == 'higru-uttr-attn':
			self.d_input = 4 * d_h1 + d_word_vec
		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h1),
			nn.Tanh()
		)

		self.final_attn = nn.Sequential(
			nn.Linear(self.d_input, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 5),
			nn.Softmax()
			)

	def uttr_attn(self, elements, combined_elements):
		n = 5


		wts = self.final_attn(combined_elements)
		
		to_return = wts[:,:,0].unsqueeze(-1).expand(-1, -1, 768) * elements[0]
		for i in range(1, n):
			to_return += wts[:,:,i].unsqueeze(-1).expand(-1, -1, 768) * elements[i]

		return to_return

	def forward(self, sents, lengths, sa_mask=None):
		"""
		:param sents: batch x seq_len x 2*d_h1
		:param lengths: numpy array 1 x batch
		:return: batch x d_h1
		"""
		w_context = self.encoder(sents, lengths)
		combined = w_context

		if self.model == 'higru-f':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			combined = [w_lcont, sents, w_rcont]
			combined = torch.cat(combined, dim=-1)
		if self.model == 'higru-sf' or self.model == 'higru-uttr-attn':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
			sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
			elements = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
			combined = torch.cat(elements, dim=-1)

		output1 = self.uttr_attn(elements, combined)
		# output1 = self.output1(combined)
		output = torch.max(output1, dim=1)[0]

		return output

# Utterance encoder with three types: higru, higru-f, and higru-sf
class UttEncoder_attn_2(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, type):
		super(UttEncoder_attn_2, self).__init__()
		self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1)
		self.d_h1, self.d_h2 = d_h1, d_h2
		self.d_input = 2 * d_h1
		self.model = type
		if self.model == 'higru-f':
			self.d_input = 2 * d_h1 + d_word_vec
		if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-2':
			self.d_input = 4 * d_h1 + d_word_vec
		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h1),
			nn.Tanh()
		)

		self.final_attn = nn.Sequential(
			nn.Linear(self.d_input, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 5),
			nn.Softmax()
			)

		self.merge_attn = nn.Sequential(
			nn.Linear(self.d_h1, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
			)

	def uttr_attn_2(self, elements, combined_elements):
		n = 5


		wts = self.final_attn(combined_elements)
		
		next_cat = wts[:,:,0].unsqueeze(-1).expand(-1, -1, self.d_h1) * elements[0]
		for i in range(1, n):
			next_cat += wts[:,:,i].unsqueeze(-1).expand(-1, -1, self.d_h1) * elements[i]

		merge_wts = self.merge_attn(next_cat)
		final_vals = merge_wts.expand(-1, -1, self.d_h1) * next_cat
		to_return = final_vals.sum(dim = 1)

		return to_return

	def forward(self, sents, lengths, sa_mask=None):
		"""
		:param sents: batch x seq_len x 2*d_h1
		:param lengths: numpy array 1 x batch
		:return: batch x d_h1
		"""
		w_context = self.encoder(sents, lengths)
		combined = w_context

		if self.model == 'higru-f':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			combined = [w_lcont, sents, w_rcont]
			combined = torch.cat(combined, dim=-1)
		if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-2':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
			sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
			elements = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
			combined = torch.cat(elements, dim=-1)

		output = self.uttr_attn_2(elements, combined)
		# output1 = self.output1(combined)
		# output = torch.max(output1, dim=1)[0]

		return output

# Utterance encoder with three types: higru, higru-f, and higru-sf
class UttEncoder_attn_3(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, type):
		super(UttEncoder_attn_3, self).__init__()
		self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1)
		self.d_h1, self.d_h2 = d_h1, d_h2
		self.d_input = 2 * d_h1
		self.model = type
		if self.model == 'higru-f':
			self.d_input = 2 * d_h1 + d_word_vec
		if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-3':
			self.d_input = 4 * d_h1 + d_word_vec
		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h1),
			nn.Tanh()
		)

		self.merge_attn = nn.Sequential(
			nn.Linear(self.d_input, 64),
			nn.ReLU(),
			nn.Linear(64, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Tanh()
			)

	def uttr_attn_3(self, combined_elements):
		merge_wts = self.merge_attn(combined_elements)
		final_vals = merge_wts.expand(-1, -1, self.d_input) * combined_elements
		to_return = final_vals.sum(dim = 1)

		return to_return

	def forward(self, sents, lengths, sa_mask=None):
		"""
		:param sents: batch x seq_len x 2*d_h1
		:param lengths: numpy array 1 x batch
		:return: batch x d_h1
		"""
		w_context = self.encoder(sents, lengths)
		combined = w_context

		if self.model == 'higru-f':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			combined = [w_lcont, sents, w_rcont]
			combined = torch.cat(combined, dim=-1)
		if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-3':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
			sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
			elements = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
			combined = torch.cat(elements, dim=-1)

		output1 = self.uttr_attn_3(combined)
		output = self.output1(output1)
		# output = torch.max(output1, dim=1)[0]

		return output

class BERT_HiGRU_base(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_base, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h1 = d_h1
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		# self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		# self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_h1, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		# self.num_outcomes = 2
		# self.classifier2 = nn.Sequential(
		# 	nn.Linear(d_h2 + feature_dim, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, self.num_outcomes)
		# )

		# self.fc_score = nn.Sequential(
		# 	nn.Linear(d_h2 + feature_dim, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, 1)
		# )


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		# s_context = self.contenc(s_embed.unsqueeze(1))[0]
		# s_context = s_context.transpose(0,1).contiguous()
		Combined = s_embed

		# if self.bidirectional==False:
		# 	if self.model == 'higru-f':
		# 		# s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_context, s_embed.unsqueeze(0)]
		# 		Combined = torch.cat(Combined, dim=-1)		
		# 	if self.model == 'higru-sf':
		# 		s_context    = s_context.squeeze(dim=0)
		# 		context_mask = get_sent_pad_attn(s_context)
		# 		SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
		# 		# SA_cont, _   = get_attention(s_context, s_context, s_context)
		# 		# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
		# 		Combined = [SA_cont, s_context, s_embed]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 		Combined = Combined.unsqueeze(dim=0)

		
		# else:
		# 	if self.model == 'higru-f':
		# 		s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 	if self.model == 'higru-sf':
		# 		s_lcont, s_rcont = s_context.chunk(2, -1)
		# 		SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
		# 		SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
		# 		Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)
		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)

		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1))
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1))
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1))
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob





# The overal HiGRU model with three types: HiGRU, HiGRU-f, HiGRU-sf
class HiGRU(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', don_model=0, feature_dim = 0):
		super(HiGRU, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding

		self.uttenc = UttEncoder(d_word_vec, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		self.feature_dim = feature_dim

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 4 * d_h2 + d_h1

		
		
		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2+feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)



	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		w_embed = self.embeddings(sents)
		# import pdb; pdb.set_trace()
		sa_mask = get_attn_pad_mask(sents, sents)
		
		s_embed = self.uttenc(w_embed, lens, sa_mask)
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			if self.model == 'higru-f':
				# s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_context, s_embed.unsqueeze(0)]
				Combined = torch.cat(Combined, dim=-1)		
			if self.model == 'higru-sf':
				# s_lcont, s_rcont = s_context.chunk(2, -1)
				s_context    = s_context.squeeze(dim=0)
				context_mask = get_sent_pad_attn(s_context)
				SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
				# SA_cont, _   = get_attention(s_context, s_context, s_context)
				# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
				Combined = [SA_cont, s_context, s_embed]
				Combined = torch.cat(Combined, dim=-1)
				Combined = Combined.unsqueeze(dim=0)

		
		else:
			if self.model == 'higru-f':
				s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
				Combined = torch.cat(Combined, dim=-1)
			if self.model == 'higru-sf':
				s_lcont, s_rcont = s_context.chunk(2, -1)
				SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
				SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
				Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
				Combined = torch.cat(Combined, dim=-1)

		# if self.bert == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		
		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			output1 = torch.cat([output1, addn_feats], dim=1)


		output  = self.classifier(output1)
		pred_scores = F.log_softmax(output, dim=1)



		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])


		'''


		return pred_scores, pred_outs, don_prob

class combo_bin(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(combo_bin, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'combo_bin':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'combo_bin':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 2)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			if self.model == 'higru-f':
				# s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_context, s_embed.unsqueeze(0)]
				Combined = torch.cat(Combined, dim=-1)		
			if self.model == 'higru-sf' or self.model == 'combo_bin':
				s_context    = s_context.squeeze(dim=0)
				context_mask = get_sent_pad_attn(s_context)
				SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
				# SA_cont, _   = get_attention(s_context, s_context, s_context)
				# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
				Combined = [SA_cont, s_context, s_embed]
				Combined = torch.cat(Combined, dim=-1)
				Combined = Combined.unsqueeze(dim=0)

		
		else:
			if self.model == 'higru-f':
				s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
				Combined = torch.cat(Combined, dim=-1)
			if self.model == 'higru-sf' or self.model == 'combo_bin':
				s_lcont, s_rcont = s_context.chunk(2, -1)
				SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
				SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
				Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
				Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)
		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)

		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class combo_multi(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(combo_multi, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'combo_multi':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'combo_multi':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words - 1
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc,  self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			if self.model == 'higru-f':
				# s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_context, s_embed.unsqueeze(0)]
				Combined = torch.cat(Combined, dim=-1)		
			if self.model == 'higru-sf' or self.model == 'combo_multi':
				s_context    = s_context.squeeze(dim=0)
				context_mask = get_sent_pad_attn(s_context)
				SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
				# SA_cont, _   = get_attention(s_context, s_context, s_context)
				# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
				Combined = [SA_cont, s_context, s_embed]
				Combined = torch.cat(Combined, dim=-1)
				Combined = Combined.unsqueeze(dim=0)

		
		else:
			if self.model == 'higru-f':
				s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
				Combined = torch.cat(Combined, dim=-1)
			if self.model == 'higru-sf' or self.model == 'combo_multi':
				s_lcont, s_rcont = s_context.chunk(2, -1)
				SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
				SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
				Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
				Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)
		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)

		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob


class BERT_HiGRU(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			if self.model == 'higru-f':
				# s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_context, s_embed.unsqueeze(0)]
				Combined = torch.cat(Combined, dim=-1)		
			if self.model == 'higru-sf':
				s_context    = s_context.squeeze(dim=0)
				context_mask = get_sent_pad_attn(s_context)
				SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
				# SA_cont, _   = get_attention(s_context, s_context, s_context)
				# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
				Combined = [SA_cont, s_context, s_embed]
				Combined = torch.cat(Combined, dim=-1)
				Combined = Combined.unsqueeze(dim=0)

		
		else:
			if self.model == 'higru-f':
				s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
				Combined = torch.cat(Combined, dim=-1)
			if self.model == 'higru-sf':
				s_lcont, s_rcont = s_context.chunk(2, -1)
				SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
				SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
				Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
				Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)
		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)

		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_LSTM(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_LSTM, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h1 = d_h1
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		# self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		# self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		# self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.wordenc = nn.GRU(768, d_h2, num_layers=1, bidirectional=True)
		# self.don_model = don_model
		self.relu = nn.ReLU()

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		# self.output1 = nn.Sequential(
		# 	nn.Linear(self.d_h1, d_h2),
		# 	nn.Tanh()
		# )
		# self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 * 2, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		# # self.num_outcomes = 2
		# self.classifier2 = nn.Sequential(
		# 	nn.Linear(d_h2 + feature_dim, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, self.num_outcomes)
		# )

		# self.fc_score = nn.Sequential(
		# 	nn.Linear(d_h2 + feature_dim, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, 1)
		# )


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		w_embed = self.wordenc(w_embed.permute(1, 0, 2))[1]
		w_embed = self.relu(torch.cat([w_embed[0], w_embed[1]], dim=-1))

		# s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		# s_embed = self.dropout_in(s_embed)  # batch x d_h1

		# s_context = self.contenc(s_embed.unsqueeze(1))[0]
		# s_context = s_context.transpose(0,1).contiguous()
		# Combined = s_context

		# if self.bidirectional==False:
		# 	if self.model == 'higru-f':
		# 		# s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_context, s_embed.unsqueeze(0)]
		# 		Combined = torch.cat(Combined, dim=-1)		
		# 	if self.model == 'higru-sf':
		# 		s_context    = s_context.squeeze(dim=0)
		# 		context_mask = get_sent_pad_attn(s_context)
		# 		SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
		# 		# SA_cont, _   = get_attention(s_context, s_context, s_context)
		# 		# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
		# 		Combined = [SA_cont, s_context, s_embed]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 		Combined = Combined.unsqueeze(dim=0)

		
		# else:
		# 	if self.model == 'higru-f':
		# 		s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 	if self.model == 'higru-sf':
		# 		s_lcont, s_rcont = s_context.chunk(2, -1)
		# 		SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
		# 		SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
		# 		Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)

		# # if self.bert_flag == True:
		# # 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		# output1 = self.output1(Combined.squeeze(0))
		# output1 = self.dropout_mid(output1)
		# if self.feature_dim > 0:
		# 	# import pdb; pdb.set_trace()
		# 	output1 = torch.cat([output1, addn_feats], dim=1)

		output  = self.classifier(w_embed)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		# sent_mask = get_sent_pad_attn(sents)
		# sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_CNN(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_CNN, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h1 = d_h1
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		kernel_map_2 = 1024
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")
			kernel_map_2 = 768

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		# self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		# self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		# self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		# self.wordenc = nn.GRU(768, d_h2, num_layers=1, bidirectional=True)
		# # self.don_model = don_model
		# self.relu = nn.ReLU()

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		# self.output1 = nn.Sequential(
		# 	nn.Linear(self.d_h1, d_h2),
		# 	nn.Tanh()
		# )
		# self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		# self.classifier = nn.Sequential(
		# 	nn.Linear(d_h2 * 2, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, self.num_classes)
		# )


		EMBED = {"char-cnn": 50, "lookup": 250} # embeddings (char-cnn, char-rnn, lookup, sae)
		NUM_FEATMAPS = 100 # feature maps generated by each kernel
		KERNEL_SIZES = [2, 3, 4]
		DROPOUT = 0.5

		self.conv = nn.ModuleList([nn.Conv2d(
		      in_channels = 1, # Ci
		      out_channels = NUM_FEATMAPS, # Co
		      kernel_size = (i, kernel_map_2) # height, width
		  ) for i in KERNEL_SIZES]) # num_kernels (K)
		self.dropout = nn.Dropout(DROPOUT)
		self.fc = nn.Linear(len(KERNEL_SIZES) * NUM_FEATMAPS, self.num_classes)
		self.softmax = nn.LogSoftmax(1)
		# # self.num_outcomes = 2
		# self.classifier2 = nn.Sequential(
		# 	nn.Linear(d_h2 + feature_dim, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, self.num_outcomes)
		# )

		# self.fc_score = nn.Sequential(
		# 	nn.Linear(d_h2 + feature_dim, d_fc),
		# 	nn.Dropout(0.5),
		# 	nn.Linear(d_fc, 1)
		# )


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		# w_embed = self.wordenc(w_embed.permute(1, 0, 2))[1]
		# w_embed = self.relu(torch.cat([w_embed[0], w_embed[1]], dim=-1))

		# s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		# s_embed = self.dropout_in(s_embed)  # batch x d_h1

		# s_context = self.contenc(s_embed.unsqueeze(1))[0]
		# s_context = s_context.transpose(0,1).contiguous()
		# Combined = s_context

		# if self.bidirectional==False:
		# 	if self.model == 'higru-f':
		# 		# s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_context, s_embed.unsqueeze(0)]
		# 		Combined = torch.cat(Combined, dim=-1)		
		# 	if self.model == 'higru-sf':
		# 		s_context    = s_context.squeeze(dim=0)
		# 		context_mask = get_sent_pad_attn(s_context)
		# 		SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
		# 		# SA_cont, _   = get_attention(s_context, s_context, s_context)
		# 		# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
		# 		Combined = [SA_cont, s_context, s_embed]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 		Combined = Combined.unsqueeze(dim=0)

		
		# else:
		# 	if self.model == 'higru-f':
		# 		s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 	if self.model == 'higru-sf':
		# 		s_lcont, s_rcont = s_context.chunk(2, -1)
		# 		SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
		# 		SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
		# 		Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)

		# # if self.bert_flag == True:
		# # 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		# output1 = self.output1(Combined.squeeze(0))
		# output1 = self.dropout_mid(output1)
		# if self.feature_dim > 0:
		# 	# import pdb; pdb.set_trace()
		# 	output1 = torch.cat([output1, addn_feats], dim=1)
		here = w_embed.unsqueeze(1)
		h = [conv(here) for conv in self.conv]
		h = [F.relu(k).squeeze(3) for k in h] # [B, Co, L] * K
		h = [F.max_pool1d(k, k.size(2)).squeeze(2) for k in h] # [B, Co] * K
		h = torch.cat(h, 1) # [B, Co * K]
		h = self.dropout(h)
		output = self.fc(h) # fully connected layer

		# output  = self.classifier(w_embed)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		# sent_mask = get_sent_pad_attn(sents)
		# sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_HiGRU_sent_attn(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_sent_attn, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= True
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-attn':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-attn':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)

		self.SA_lcont_attn = nn.Sequential(
			nn.Linear(self.d_h2, 1),
			nn.Sigmoid()
			)

		self.s_lcont_attn = nn.Sequential(
			nn.Linear(self.d_h2, 1),
			nn.Sigmoid()
			)

		self.s_rcont_attn = nn.Sequential(
			nn.Linear(self.d_h2, 1),
			nn.Sigmoid()
			)

		self.SA_lcont_attn = nn.Sequential(
			nn.Linear(self.d_h2, 1),
			nn.Sigmoid()
			)

		self.s_embed_attn = nn.Sequential(
			nn.Linear(d_h1, 1),
			nn.Sigmoid()
			)

	def higru_attn(self, elements):
		SA_lcont, s_lcont, s_embed, s_rcont, SA_rcont = elements
		
		SA_lcont_new = self.SA_lcont_attn(SA_lcont.squeeze(0)) * SA_lcont
		s_lcont_new = self.s_lcont_attn(s_lcont.squeeze(0)) * s_lcont
		s_embed_new = self.s_embed_attn(s_embed.squeeze(0)) * s_embed
		s_rcont_new = self.s_rcont_attn(s_rcont.squeeze(0)) * s_rcont
		SA_rcont_new = self.SA_lcont_attn(SA_rcont.squeeze(0)) * SA_rcont
		
		return [SA_lcont_new, s_lcont_new, s_embed_new, s_rcont_new, SA_rcont_new]


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			s_context    = s_context.squeeze(dim=0)
			context_mask = get_sent_pad_attn(s_context)
			SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
			# SA_cont, _   = get_attention(s_context, s_context, s_context)
			# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
			Combined = [SA_cont, s_context, s_embed]
			Combined = self.higru_attn(Combined)
			Combined = Combined.unsqueeze(dim=0)

		
		else:
			s_lcont, s_rcont = s_context.chunk(2, -1)
			SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
			SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
			Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
			Combined = self.higru_attn(Combined)
			Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)
		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)


		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_HiGRU_sent_attn_2(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_sent_attn_2, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		self.d_h1 = d_h1
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f' or self.model == 'higru-sent-attn-2-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-attn-2':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f' or self.model == 'higru-sent-attn-2-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-attn-2':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim
		self.feature_dim = feature_dim

		self.output1 = nn.Sequential(
			nn.Linear(512, d_h2 + feature_dim),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.output1_gru = nn.GRU(self.d_input + 512 + feature_dim, 512)
		
		self.relu_gru = nn.ReLU()

		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)

		self.final_attn = nn.Sequential(
			nn.Linear(self.d_input, 2048),
			nn.ReLU(),
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
			nn.Tanh()
			)

		self.higru_sent_attn = nn.Sequential(
			nn.Linear(self.d_input + self.feature_dim + 512, 2048),
			nn.ReLU(),
			nn.Linear(2048, 2048),
			nn.ReLU(),
			nn.Linear(2048, 1),
			nn.Tanh()
			)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)
		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context


		if self.bidirectional==False:
			if '-f' in self.model:
				# s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_context, s_embed.unsqueeze(0)]
				Combined = torch.cat(Combined, dim=-1).squeeze(0)
			else:
				s_context    = s_context.squeeze(dim=0)
				context_mask = get_sent_pad_attn(s_context)
				SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
				# SA_cont, _   = get_attention(s_context, s_context, s_context)
				# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
				Combined = [SA_cont, s_context, s_embed]
				Combined = torch.cat(Combined, dim=-1).squeeze(0)
				# Combined = Combined.unsqueeze(dim=0)

		else:
			if '-f' in self.model:
				s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
				Combined = torch.cat(Combined, dim=-1).squeeze(0)
			else:
				s_lcont, s_rcont = s_context.chunk(2, -1)
				SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
				SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
				Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
				Combined = torch.cat(Combined, dim=-1).squeeze(0)

		results = torch.zeros((Combined.shape[0], self.num_classes)).cuda()
		context = torch.zeros((1, 1, 512)).cuda()
		for i in range(Combined.shape[0]):
			vec_here = Combined[i, :]
			total_here = torch.cat([vec_here, context.squeeze(0).squeeze(0)], dim = -1)
			if self.feature_dim > 0:
				# import pdb; pdb.set_trace()
				total_here = torch.cat([total_here, addn_feats[i,:]], dim=-1)
			attn_vec = self.higru_sent_attn(total_here) * total_here
			out_here, _ = self.output1_gru(attn_vec.unsqueeze(0).unsqueeze(0), context)
			out_here = self.relu_gru(out_here)
			context = out_here
			output1 = self.output1(out_here)
			output1 = self.dropout_mid(output1)

			# if self.bert_flag == True:
			# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

			output  = self.classifier(output1.squeeze(0))
			results[i, :] = output

		log_pred_scores = F.log_softmax(results, dim=1)
		pred_scores = F.softmax(results, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		# sent_mask = get_sent_pad_attn(sents)
		# sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_HiGRU_sent_attn_mask(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_sent_attn_mask, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-attn-mask':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-attn-mask':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim
		self.feature_dim = feature_dim

		self.output1 = nn.Sequential(
			nn.Linear(512, d_h2 + feature_dim),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.output1_gru = nn.GRU(self.d_input + 512 + feature_dim, 512)
		
		self.relu_gru = nn.ReLU()

		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)

		self.final_attn = nn.Sequential(
			nn.Linear(self.d_input, 2048),
			nn.ReLU(),
			nn.Linear(2048, 512),
			nn.ReLU(),
			nn.Linear(512, 1),
			nn.Tanh()
			)

		self.higru_sent_attn = nn.Sequential(
			nn.Linear(self.d_input + self.feature_dim + 512, 2048),
			nn.ReLU(),
			nn.Linear(2048, 2048),
			nn.ReLU(),
			nn.Linear(2048, 1),
			nn.Tanh()
			)


	def forward(self, sents, lens, addn_feats, mask):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)
		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			s_context    = s_context.squeeze(dim=0)
			context_mask = get_sent_pad_attn(s_context)
			SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
			# SA_cont, _   = get_attention(s_context, s_context, s_context)
			# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
			Combined = [SA_cont, s_context, s_embed]
			Combined = torch.cat(Combined, dim = -1)

		
		else:
			s_lcont, s_rcont = s_context.chunk(2, -1)
			SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
			SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
			Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
			Combined = torch.cat(Combined, dim=-1).squeeze(0)

		results = torch.zeros((Combined.shape[0], self.num_classes)).cuda()
		context = torch.zeros((1, 1, 512)).cuda()
		for i in range(Combined.shape[0]):
			vec_here = Combined[i, :]
			total_here = torch.cat([vec_here, context.squeeze(0).squeeze(0)], dim = -1)
			if self.feature_dim > 0:
				# import pdb; pdb.set_trace()
				total_here = torch.cat([total_here, addn_feats[i,:]], dim=-1)
			
			attn_vec = self.higru_sent_attn(total_here) * total_here
			out_here, _ = self.output1_gru(attn_vec.unsqueeze(0).unsqueeze(0), context)
			out_here = self.relu_gru(out_here)
			if mask[i] == 1:
				context = out_here
			output1 = self.output1(out_here)
			output1 = self.dropout_mid(output1)

			# if self.bert_flag == True:
			# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

			output  = self.classifier(output1.squeeze(0))
			results[i, :] = output

		log_pred_scores = F.log_softmax(results, dim=1)
		pred_scores = F.softmax(results, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		# sent_mask = get_sent_pad_attn(sents)
		# sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob



class BERT_HiGRU_sent_conn_mask(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_sent_conn_mask, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-conn-mask':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-sent-conn-mask':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim + self.num_classes, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)





	def forward(self, sents, lens, addn_feats, mask):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""

		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			if self.model == 'higru-f':
				# s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_context, s_embed.unsqueeze(0)]
				Combined = torch.cat(Combined, dim=-1)		
			if self.model == 'higru-sf' or self.model == 'higru-sent-conn-mask':
				s_context    = s_context.squeeze(dim=0)
				context_mask = get_sent_pad_attn(s_context)
				SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
				# SA_cont, _   = get_attention(s_context, s_context, s_context)
				# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
				Combined = [SA_cont, s_context, s_embed]
				Combined = torch.cat(Combined, dim=-1)
				Combined = Combined.unsqueeze(dim=0)

		
		else:
			if self.model == 'higru-f':
				s_lcont, s_rcont = s_context.chunk(2,-1)
				Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
				Combined = torch.cat(Combined, dim=-1)
			if self.model == 'higru-sf' or self.model == 'higru-sent-conn-mask':
				s_lcont, s_rcont = s_context.chunk(2, -1)
				SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
				SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
				Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
				Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)
		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)

		context = torch.zeros((1, self.num_classes)).cuda()
		output = torch.zeros((output1.shape[0], self.num_classes)).cuda()

		for i in range(output1.shape[0]):
			vec_here = output1[i,:]
			vec_here = torch.cat([vec_here.unsqueeze(0), context], dim = -1)
			here = self.classifier(vec_here)
			output[i, :] = here
			if mask[i] == 1:
				context = F.softmax(here, dim = -1)

		# output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_HiGRU_uttr_attn(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_uttr_attn, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder_attn(self.bert_emb_dim, d_h1, d_h2, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= True
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-uttr-attn':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-uttr-attn':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			s_context    = s_context.squeeze(dim=0)
			context_mask = get_sent_pad_attn(s_context)
			SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
			# SA_cont, _   = get_attention(s_context, s_context, s_context)
			# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
			Combined = [SA_cont, s_context, s_embed]
			Combined = torch.cat(Combined, dim = -1)
			Combined = Combined.unsqueeze(dim=0)

		
		else:
			s_lcont, s_rcont = s_context.chunk(2, -1)
			SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
			SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
			Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
			Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)
		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)


		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob


class BERT_HiGRU_uttr_attn_2(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_uttr_attn_2, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder_attn_2(self.bert_emb_dim, d_h1, d_h2, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= True
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-2':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-2':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			s_context    = s_context.squeeze(dim=0)
			context_mask = get_sent_pad_attn(s_context)
			SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
			# SA_cont, _   = get_attention(s_context, s_context, s_context)
			# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
			Combined = [SA_cont, s_context, s_embed]
			Combined = torch.cat(Combined, dim = -1)
			Combined = Combined.unsqueeze(dim=0)

		
		else:
			s_lcont, s_rcont = s_context.chunk(2, -1)
			SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
			SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
			Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
			Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)
		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)


		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob

class BERT_HiGRU_uttr_attn_3(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', bert_flag=False, don_model=0, trainable= False, feature_dim= 0, long_bert = 0):
		super(BERT_HiGRU_uttr_attn_3, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim=1024
		else:
			self.bert_emb_dim=768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim
		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad = trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder_attn_3(self.bert_emb_dim, d_h1, d_h2, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= True
		# self.bert_flag= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model

		if self.bidirectional==False:
			self.d_input= d_h2
			if self.model == 'higru-f':
				self.d_input = d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-3':
				self.d_input = 2 * d_h2 + d_h1

		else:
			self.d_input = 2 * d_h2
			if self.model == 'higru-f':
				self.d_input = 2 * d_h2 + d_h1
			if self.model == 'higru-sf' or self.model == 'higru-uttr-attn-3':
				self.d_input = 4 * d_h2 + d_h1

		# if self.bert_flag:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		self.output1 = nn.Sequential(
			nn.Linear(self.d_input, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats = None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		s_context = s_context.transpose(0,1).contiguous()
		Combined = s_context

		if self.bidirectional==False:
			s_context    = s_context.squeeze(dim=0)
			context_mask = get_sent_pad_attn(s_context)
			SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
			# SA_cont, _   = get_attention(s_context, s_context, s_context)
			# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
			Combined = [SA_cont, s_context, s_embed]
			Combined = torch.cat(Combined, dim = -1)
			Combined = Combined.unsqueeze(dim=0)

		
		else:
			s_lcont, s_rcont = s_context.chunk(2, -1)
			SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
			SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
			Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
			Combined = torch.cat(Combined, dim=-1)

		# if self.bert_flag == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)
		output1 = self.output1(Combined.squeeze(0))
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			# import pdb; pdb.set_trace()
			output1 = torch.cat([output1, addn_feats], dim=1)


		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''
		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob


class BiGRU(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru', don_model=0, feature_dim =0):
		super(BiGRU, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		
		# load word2vec
		self.embeddings = embedding

		self.uttenc = UttEncoder(d_word_vec, d_h1, self.model)
		self.dropout_in = nn.Dropout(0.5)

		self.bidirectional= False
		self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=self.bidirectional)
		self.don_model = don_model
		self.feature_dim = feature_dim

		# if self.bidirectional==False:
		# 	self.d_input= d_h2
		# 	if self.model == 'higru-f':
		# 		self.d_input = d_h2 + d_h1
		# 	if self.model == 'higru-sf':
		# 		self.d_input = 2 * d_h2 + d_h1

		# else:
		# 	self.d_input = 2 * d_h2
		# 	if self.model == 'higru-f':
		# 		self.d_input = 2 * d_h2 + d_h1
		# 	if self.model == 'higru-sf':
		# 		self.d_input = 4 * d_h2 + d_h1

		# if self.bert:
		# 	self.d_input= self.d_input+ self.bert_emb_dim

		
		self.output1 = nn.Sequential(
			nn.Linear(d_h1, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2 + feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats=None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		sa_mask = get_attn_pad_mask(sents, sents)
		w_embed = self.embeddings(sents)

		s_embed = self.uttenc(w_embed, lens, sa_mask)
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		# s_context = self.contenc(s_embed.unsqueeze(1))[0]
		# s_context = s_context.transpose(0,1).contiguous()
		# Combined = s_context

		# if self.bidirectional==False:
		# 	if self.model == 'higru-f':
		# 		# s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_context, s_embed.unsqueeze(0)]
		# 		Combined = torch.cat(Combined, dim=-1)		
		# 	if self.model == 'higru-sf':
		# 		# s_lcont, s_rcont = s_context.chunk(2, -1)
		# 		s_context    = s_context.squeeze(dim=0)
		# 		context_mask = get_sent_pad_attn(s_context)
		# 		SA_cont, _   = get_sent_attention(s_context, s_context,s_context, context_mask)
		# 		# SA_cont, _   = get_attention(s_context, s_context, s_context)
		# 		# Combined = [SA_cont, s_context, s_embed.unsqueeze(0)]
		# 		Combined = [SA_cont, s_context, s_embed]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 		Combined = Combined.unsqueeze(dim=0)

		
		# else:
		# 	if self.model == 'higru-f':
		# 		s_lcont, s_rcont = s_context.chunk(2,-1)
		# 		Combined = [s_lcont, s_embed.unsqueeze(0), s_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)
		# 	if self.model == 'higru-sf':
		# 		s_lcont, s_rcont = s_context.chunk(2, -1)
		# 		SA_lcont, _ = get_attention(s_lcont, s_lcont, s_lcont)
		# 		SA_rcont, _ = get_attention(s_rcont, s_rcont, s_rcont)
		# 		Combined = [SA_lcont, s_lcont, s_embed.unsqueeze(0), s_rcont, SA_rcont]
		# 		Combined = torch.cat(Combined, dim=-1)

		# if self.bert == True:
		# 	Combined= torch.cat([Combined,bert_emb.unsqueeze(0)], dim=-1)

		
		output1 = self.output1(s_embed)
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			output1 = torch.cat([output1, addn_feats], dim=1)

		output  = self.classifier(output1)
		pred_scores = F.log_softmax(output, dim=1)


		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''

		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		
		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])


		'''


		return pred_scores, pred_outs, don_prob



class BERT_BiGRU(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_fc, emodict, worddict, embedding, type='higru',  don_model=0, trainable= False, feature_dim=0):
		super(BERT_BiGRU, self).__init__()
		self.model = type
		self.max_length = worddict.max_length
		self.max_dialog = worddict.max_dialog
		self.d_h2 = d_h2
		if long_bert == 2:
			self.bert_emb_dim = 1024
		else:
			self.bert_emb_dim = 768
		# load word2vec
		self.embeddings = embedding
		self.feature_dim = feature_dim

		from transformers import BertModel
		if long_bert == 2:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			print("Large BERT chosen")
		if long_bert == 1:
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			print("Base BERT chosen")

		for p in self.bert.parameters():
			p.requires_grad= trainable
			# if trainable == 1:
			# 	p.requires_grad = True
			# if trainable == 0:
			# 	p.requires_grad = False

		self.uttenc = UttEncoder(self.bert_emb_dim, d_h1, self.model)
		self.don_model = don_model
		self.dropout_in = nn.Dropout(0.5)
		
		self.output1 = nn.Sequential(
			nn.Linear(d_h1, d_h2),
			nn.Tanh()
		)
		self.dropout_mid = nn.Dropout(0.5)

		self.num_classes = emodict.n_words
		self.classifier = nn.Sequential(
			nn.Linear(d_h2+ feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_classes)
		)

		self.num_outcomes = 2
		self.classifier2 = nn.Sequential(
			nn.Linear(d_h2+ feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, self.num_outcomes)
		)

		self.fc_score = nn.Sequential(
			nn.Linear(d_h2+ feature_dim, d_fc),
			nn.Dropout(0.5),
			nn.Linear(d_fc, 1)
		)


	def forward(self, sents, lens, addn_feats=None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)


		
		# w_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		bert_sa_mask = get_word_pad_attns(sents)

		# import pdb; pdb.set_trace()
		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

		w_embed = outputs[0]
		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
		s_embed = self.dropout_in(s_embed)  # batch x d_h1

		
		output1 = self.output1(s_embed.squeeze(0))
		output1 = self.dropout_mid(output1)

		if self.feature_dim > 0:
			output1 = torch.cat([output1, addn_feats], dim=1)


		output  = self.classifier(output1)
		log_pred_scores = F.log_softmax(output, dim=1)
		pred_scores = F.softmax(output, dim=1)

		# pred_scores = output
		# computes the sentence mask of the attention, essentially creating a lower traingular matrix.
		sent_mask = get_sent_pad_attn(sents)
		sent_output, sent_attn =  get_sent_attention(output1, output1, output1, sent_mask)

		output2  = None
		pred_outs = None
		don_prob  = None

		'''

		if self.don_model == 0:   # mask last leg, consider only the last hidden stage
			output2  = self.classifier2(output1)
			pred_outs= F.log_softmax(output2, dim=1)
 
		if self.don_model == 1:   # do self attention on the hidden states, mask the last one only
			output2  = self.classifier2(sent_output)
			pred_outs= F.log_softmax(output2, dim=1)

		if self.don_model == 2:  # do some smoothing over the preds 
			output2     = self.classifier2(sent_output)
			# pred_outs   = F.log_softmax(output2, dim=1)
			
			outs = F.softmax(output2, dim =1)
			don_prob = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = outs[0][1]

			for i in range(1, len(outs)):
				don_prob[i] = 0.5*don_prob[i-1]+ 0.5*outs[i][1]

		if self.don_model==3:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = tanh(0.5*don_prob[i-1]+ 0.5* outs[i])
		

		if self.don_model==4:
			output2     = self.fc_score(sent_output)
			tanh        = torch.nn.Tanh()

			outs        = tanh(output2)
			don_prob    = torch.zeros((len(outs),1)).cuda(sents.device)
			don_prob[0] = torch.sigmoid(outs[0])

			for i in range(1,len(outs)):
				don_prob[i] = torch.sigmoid(don_prob[i-1]+outs[i])

		'''
		return log_pred_scores, pred_outs, don_prob
