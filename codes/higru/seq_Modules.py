import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import Const

import torch
from torch import nn
import torch.nn.functional as F

import random, math

def mask_(matrices, maskval=0.0, mask_diagonal=True):
	"""
	Masks out all values in the given batch of matrices where i <= j holds,
	i < j if mask_diagonal is false
	In place operation
	:param tns:
	:return:
	"""

	b, h, w = matrices.size()

	indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1)
	matrices[:, indices[0], indices[1]] = maskval

class SelfAttentionWide(nn.Module):
	def __init__(self, emb, heads=8, mask=False):
		"""
		:param emb:
		:param heads:
		:param mask:
		"""

		super().__init__()

		self.emb = emb
		self.heads = heads
		self.mask = mask

		self.tokeys = nn.Linear(emb, emb * heads, bias=False)
		self.toqueries = nn.Linear(emb, emb * heads, bias=False)
		self.tovalues = nn.Linear(emb, emb * heads, bias=False)

		self.unifyheads = nn.Linear(heads * emb, emb)

	def forward(self, x):

		b, t, e = x.size()
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

		keys    = self.tokeys(x)   .view(b, t, h, e)
		queries = self.toqueries(x).view(b, t, h, e)
		values  = self.tovalues(x) .view(b, t, h, e)

		# compute scaled dot-product self-attention

		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
		values = values.transpose(1, 2).contiguous().view(b * h, t, e)

		queries = queries / (e ** (1/4))
		keys    = keys / (e ** (1/4))
		# - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
		#   This should be more memory efficient

		# - get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))

		assert dot.size() == (b*h, t, t)

		if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
			mask_(dot, maskval=float('-inf'), mask_diagonal=False)

		dot = F.softmax(dot, dim=2)
		# - dot now has row-wise self-attention probabilities

		# apply the self attention to the values
		out = torch.bmm(dot, values).view(b, h, t, e)

		# swap h, t back, unify heads
		out = out.transpose(1, 2).contiguous().view(b, t, h * e)

		return self.unifyheads(out)


class SelfAttentionNarrow(nn.Module):

	def __init__(self, emb, heads=8, mask=False):
		"""
		:param emb:
		:param heads:
		:param mask:
		"""

		super().__init__()

		assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

		self.emb = emb
		self.heads = heads
		self.mask = mask

		s = emb // heads
		# - We will break the embedding into `heads` chunks and feed each to a different attention head

		self.tokeys    = nn.Linear(s, s, bias=False)
		self.toqueries = nn.Linear(s, s, bias=False)
		self.tovalues  = nn.Linear(s, s, bias=False)

		self.unifyheads = nn.Linear(heads * s, emb)

	def forward(self, x):

		b, t, e = x.size()
		h = self.heads
		assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

		s = e // h
		x = x.view(b, t, h, s)

		keys    = self.tokeys(x)
		queries = self.toqueries(x)
		values  = self.tovalues(x)

		assert keys.size() == (b, t, h, s)
		assert queries.size() == (b, t, h, s)
		assert values.size() == (b, t, h, s)

		# Compute scaled dot-product self-attention

		# - fold heads into the batch dimension
		keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
		queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
		values = values.transpose(1, 2).contiguous().view(b * h, t, s)

		queries = queries / (e ** (1/4))
		keys    = keys / (e ** (1/4))
		# - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
		#   This should be more memory efficient

		# - get dot product of queries and keys, and scale
		dot = torch.bmm(queries, keys.transpose(1, 2))

		assert dot.size() == (b*h, t, t)

		if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
			mask_(dot, maskval=float('-inf'), mask_diagonal=False)

		dot = F.softmax(dot, dim=2)
		# - dot now has row-wise self-attention probabilities

		# apply the self attention to the values
		out = torch.bmm(dot, values).view(b, h, t, s)

		# swap h, t back, unify heads
		out = out.transpose(1, 2).contiguous().view(b, t, s * h)

		return self.unifyheads(out)

class TransformerBlock(nn.Module):

	def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0, wide=True):
		super().__init__()

		self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
					else SelfAttentionNarrow(emb, heads=heads, mask=mask)
		self.mask = mask

		self.norm1 = nn.LayerNorm(emb)
		self.norm2 = nn.LayerNorm(emb)

		self.ff = nn.Sequential(
			nn.Linear(emb, ff_hidden_mult * emb),
			nn.ReLU(),
			nn.Linear(ff_hidden_mult * emb, emb)
		)

		self.do = nn.Dropout(dropout)

	def forward(self, x):

		attended = self.attention(x)

		x = self.norm1(attended + x)

		x = self.do(x)

		fedforward = self.ff(x)

		x = self.norm2(fedforward + x)

		x = self.do(x)

		return x


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

def get_sent_pad_attn(seq_q):
	pad_sent_mask = np.zeros((seq_q.shape[0], seq_q.shape[0]))
	for i in range(seq_q.shape[0]):
		pad_sent_mask[i,:(i+1)] =     1
		pad_sent_mask[i,i+1:]   = -1e10

	pad_sent_mask = torch.FloatTensor(pad_sent_mask).cuda(seq_q.device)
	return pad_sent_mask


# Pad for utterances with variable lengths and maintain the order of them after GRU
class GRUencoder(nn.Module):
	def __init__(self, d_emb, d_out, num_layers, encoder_type='GRU', bidirectional=False):
		super(GRUencoder, self).__init__()
		# default encoder 2 layers

		# add_condition about encoder_type later once this works...

		if encoder_type =='GRU':
			self.gru = nn.GRU(input_size=d_emb, hidden_size=d_out,
							  bidirectional=bidirectional, num_layers=num_layers, dropout=0.3)
		elif encoder_type =='LSTM':
			self.gru = nn.LSTM(input_size=d_emb, hidden_size=d_out,
							  bidirectional=bidirectional, num_layers=num_layers, dropout=0.3)

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

class UttEncoder(nn.Module):
	def __init__(self, d_word_vec, d_h1, args, encoder_type='GRU', bidirectional = False):
		super(UttEncoder, self).__init__()
		self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1, encoder_type=encoder_type, bidirectional=bidirectional)

		dir_num=1
		if bidirectional:
			dir_num=2
		self.d_input = dir_num * d_h1

		self.model = args.text_model
		if self.model == 'higru-f':
			self.d_input = dir_num * d_h1 + d_word_vec
		if self.model == 'higru-sf':
			self.d_input = 2*dir_num * d_h1 + d_word_vec

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

		if self.model == 'higru-f':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			combined = [w_lcont, sents, w_rcont]
			combined = torch.cat(combined, dim=-1)
		if self.model == 'higru-sf':
			w_lcont, w_rcont = w_context.chunk(2, -1)
			sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
			sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
			combined = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
			combined = torch.cat(combined, dim=-1)

		output1 = self.output1(combined)
		output = torch.max(output1, dim=1)[0]

		return output


# class SeqEncoder(nn.Module):
# 	def __init__(self, d_h1, tagidx, args):
# 		super(SeqEncoder, self).__init__()
# 		self.d_input = d_h1
# 		self.embeddings = nn.Embedding(len(tagidx), d_h1)
# 		torch.nn.init.uniform_(self.embeddings.weight, -0.25, 0.25)
# 		self.embeddings.weight.requires_grad = True

# 		self.encoder = GRUencoder(d_h1, d_h1, num_layers=1, encoder_type=args.model_type)

# 		self.output1 = nn.Sequential(
# 			nn.Linear(d_h1, d_h1),
# 			nn.Tanh()
# 		)
# 		self.dropout_mid = nn.Dropout(0.5)

		
# 		self.num_classes = 2

# 		self.classifier = nn.Sequential(
# 			nn.Linear(d_h1, d_h1),
# 			nn.Dropout(0.5),
# 			nn.Linear(d_h1, self.num_classes)
# 		)

# 		self.pool_type = args.pool_type


# 	def forward(self, sents, lengths):
# 		"""
# 		:param sents: batch x seq_len x 2*d_h1
# 		:param lengths: numpy array 1 x batch
# 		:return: batch x d_h1
# 		"""
# 		if len(sents.size()) < 2:
# 			sents = sents.unsqueeze(0)

# 		utt_embed = self.embeddings(sents)
# 		sa_mask = get_attn_pad_mask(sents, sents)
		
# 		# s_embed = self.uttenc(w_embed, lens, sa_mask)
# 		# s_embed = self.dropout_in(s_embed)  # batch x d_h1
# 		combined = self.encoder(utt_embed, lengths)

# 		output1  = self.output1(combined)
		
# 		if self.pool_type =='max':
# 			output1  = torch.max(output1, dim=1)[0]
# 		elif self.pool_type == 'last':
# 			output1  = output1[:,-1,:]
# 		elif self.pool_type == 'mean':
# 			output1  = torch.mean(output1, dim=1)[0]

# 		output1  = self.dropout_mid(output1)
# 		output   = self.classifier(output1)
# 		pred_scores = F.log_softmax(output, dim=1)

# 		return pred_scores


# class TextEncoder(nn.Module):
# 	def __init__(self, d_word_vec, d_h1, d_h2, args, bert_type='base', encoder_type='GRU', trainable= False):
# 		super(TextEncoder, self).__init__()

		
		
# 		from transformers import BertModel
# 		if bert_type =='base':
# 			self.bert = BertModel.from_pretrained('bert-base-uncased')
# 			self.bert_emb_dim=768
# 		else:
# 			self.bert = BertModel.from_pretrained('bert-large-uncased')
# 			self.bert_emb_dim=768

# 		for p in self.bert.parameters():
# 			p.requires_grad = trainable
			
# 		self.uttenc  = UttEncoder(self.bert_emb_dim, d_h1, args,encoder_type= encoder_type, bidirectional=True)

# 		if encoder_type =='GRU':
# 			self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=False)
# 		else:
# 			self.contenc = nn.LSTM(d_h1, d_h2, num_layers=1, bidirectional=False)


# 		self.dropout_in = nn.Dropout(0.1)

# 		self.output1 = nn.Sequential(
# 			nn.Linear(d_h2, d_h2),
# 			nn.Tanh()
# 		)
# 		self.dropout_mid = nn.Dropout(0.1)

		
# 		self.num_classes = 2

# 		self.classifier = nn.Sequential(
# 			nn.Linear(d_h2, d_h2),
# 			nn.Dropout(0.1),
# 			nn.Linear(d_h2, self.num_classes)
# 		)

# 		self.pool_type = args.pool_type


# 	def forward(self, sents, lens):
# 		"""
# 		:param sents: batch x seq_len
# 		:param lens: 1 x batch
# 		:return:
# 		"""
		
# 		if len(sents.size()) < 2:
# 			sents = sents.unsqueeze(0)

# 		# w_embed = self.embeddings(sents)
# 		sa_mask = get_attn_pad_mask(sents, sents)
		
# 		bert_sa_mask = get_word_pad_attns(sents)

# 		# import pdb; pdb.set_trace()
# 		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
# 		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
# 		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

# 		w_embed = outputs[0]
# 		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
# 		s_embed = self.dropout_in(s_embed)  # batch x d_h1

# 		s_context = self.contenc(s_embed.unsqueeze(1))[0]
# 		combined = s_context.transpose(0,1).contiguous()
		
# 		output1  = self.output1(combined)
		
# 		if self.pool_type =='max':
# 			output1  = torch.max(output1, dim=1)[0]
# 		elif self.pool_type == 'last':
# 			output1  = output1[:,-1,:]
# 		elif self.pool_type == 'mean':
# 			output1  = torch.mean(output1, dim=1)[0]

# 		output1  = self.dropout_mid(output1)
# 		output   = self.classifier(output1)
# 		pred_scores = F.log_softmax(output, dim=1)

# 		return pred_scores
		

# class TextSeqEncoder(nn.Module):
# 	def __init__(self, d_word_vec, d_h1, d_h2, d_tag, args, tagidx, bert_type='base', encoder_type='GRU', trainable= False):
# 		super(TextSeqEncoder, self).__init__()

		
		
# 		from transformers import BertModel
# 		if bert_type =='base':
# 			self.bert = BertModel.from_pretrained('bert-base-uncased')
# 			self.bert_emb_dim=768
# 		else:
# 			self.bert = BertModel.from_pretrained('bert-large-uncased')
# 			self.bert_emb_dim=768

# 		for p in self.bert.parameters():
# 			p.requires_grad = trainable
			
# 		self.uttenc  = UttEncoder(self.bert_emb_dim, d_h1, args,encoder_type= encoder_type, bidirectional=True)

# 		if encoder_type =='GRU':
# 			self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=False)
# 		else:
# 			self.contenc = nn.LSTM(d_h1, d_h2, num_layers=1, bidirectional=False)


# 		self.dropout_small = nn.Dropout(0.1)

# 		self.output1 = nn.Sequential(
# 			nn.Linear(d_h2, d_h2),
# 			nn.Tanh()
# 		)
# 		self.dropout_mid = nn.Dropout(0.5)


# 		self.embeddings = nn.Embedding(len(tagidx), d_tag)
# 		torch.nn.init.uniform_(self.embeddings.weight, -0.25, 0.25)
# 		self.embeddings.weight.requires_grad = True

# 		self.encoder = GRUencoder(d_tag, d_tag, num_layers=1, encoder_type=args.model_type)

# 		self.output2 = nn.Sequential(
# 			nn.Linear(d_tag, d_tag),
# 			nn.Tanh()
# 		)

		
# 		self.num_classes = 2

# 		self.classifier = nn.Sequential(
# 			nn.Linear(d_h2+d_tag, d_h2+d_tag),
# 			nn.Dropout(0.1),
# 			nn.Linear(d_h2+d_tag, self.num_classes)
# 		)

# 		self.pool_type = args.pool_type


# 	def forward(self, sents, lens, tags, taglens):
# 		"""
# 		:param sents: batch x seq_len
# 		:param lens: 1 x batch
# 		:return:
# 		"""
# 		# import pdb; pdb.set_trace()

# 		if len(sents.size()) < 2:
# 			sents = sents.unsqueeze(0)

# 		# w_embed = self.embeddings(sents)
# 		sa_mask = get_attn_pad_mask(sents, sents)
		
# 		bert_sa_mask = get_word_pad_attns(sents)

# 		# import pdb; pdb.set_trace()
# 		outputs = self.bert(input_ids = sents, attention_mask = bert_sa_mask, token_type_ids=None, position_ids= None, head_mask= None, inputs_embeds= None)
# 		# s_embed = outputs[1]    # This is the BERT sentence embedding              # BERT CLS TOKEN
# 		# w_embed = self.bert_dropout(outputs[0]) # This is the bert word embeddings

# 		w_embed = outputs[0]
# 		s_embed = self.uttenc(w_embed, lens, sa_mask)           # HIGRU utterance encoder
# 		s_embed = self.dropout_small(s_embed)  # batch x d_h1

# 		s_context = self.contenc(s_embed.unsqueeze(1))[0]
# 		s_context = s_context.transpose(0,1).contiguous()
# 		output1  = self.output1(s_context)

# 		# output1  = self.dropout_small(output1)

# 		if len(tags.size()) < 2:
# 			tags = tags.unsqueeze(0)

# 		tag_embed = self.embeddings(tags)
# 		tag_mask = get_attn_pad_mask(tags, tags)

# 		tag_context = self.encoder(tag_embed, taglens)
# 		output2     = self.output2(tag_context)

# 		# output2     = self.dropout_mid(output2)

# 		combined = torch.cat([output1,output2], dim=2)

# 		if self.pool_type =='max':
# 			combined  = torch.max(combined, dim=1)[0]
# 		elif self.pool_type == 'last':
# 			combined  = combined[:,-1,:]
# 		elif self.pool_type == 'mean':
# 			combined  = torch.mean(combined, dim=1)[0]

# 		combined = self.dropout_mid(combined)
# 		output   = self.classifier(combined)
# 		pred_scores = F.log_softmax(output, dim=1)

# 		return pred_scores
		

# class CTransformer(nn.Module):
# 	"""
# 	Transformer for classifying sequences
# 	"""

# 	def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, args, dropout=0.0, wide=False):
# 		"""
# 		:param emb: Embedding dimension
# 		:param heads: nr. of attention heads
# 		:param depth: Number of transformer blocks
# 		:param seq_length: Expected maximum sequence length
# 		:param num_tokens: Number of tokens (usually words) in the vocabulary
# 		:param num_classes: Number of classes.
# 		:param max_pool: If true, use global max pooling in the last layer. If false, use global
# 						 average pooling.
# 		"""
# 		super().__init__()

# 		self.num_tokens = num_tokens

# 		self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
# 		self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

# 		tblocks = []
# 		for i in range(depth):
# 			tblocks.append(
# 				TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

# 		self.tblocks = nn.Sequential(*tblocks)

# 		self.pool_type = args.pool_type

# 		self.toprobs = nn.Linear(emb, num_classes)

# 		self.do = nn.Dropout(dropout)

# 	def forward(self, x):
# 		"""
# 		:param x: A batch by sequence length integer tensor of token indices.
# 		:return: predicted log-probability vectors for each token based on the preceding tokens.
# 		"""
# 		tokens = self.token_embedding(x)
# 		if len(tokens.size()) <=2:
# 			tokens = tokens.unsqueeze(0)

# 		b, t, e = tokens.size()

# 		positions = self.pos_embedding(torch.arange(t, device=x.device))[None, :, :].expand(b, t, e)
# 		x = tokens + positions
# 		x = self.do(x)

# 		x = self.tblocks(x)

# 		x = x.max(dim=1)[0] if self.pool_type=='max' else x.mean(dim=1) # pool over the time dimension

# 		x = self.toprobs(x)

# 		return F.log_softmax(x, dim=1)



class SeqEncoderOnly(nn.Module):
	def __init__(self, d_h1, tagidx, args):
		super(SeqEncoderOnly, self).__init__()
		self.d_input = d_h1
		self.embeddings = nn.Embedding(len(tagidx), d_h1)
		torch.nn.init.uniform_(self.embeddings.weight, -0.25, 0.25)
		self.embeddings.weight.requires_grad = True

		self.encoder = GRUencoder(d_h1, d_h1, num_layers=1, encoder_type=args.model_type)

		self.output1 = nn.Sequential(
			nn.Linear(d_h1, d_h1),
			nn.Tanh()
		)

		self.dropout_small = nn.Dropout(0.1)
		self.dropout_mid   = nn.Dropout(0.5)
		self.pool_type     = args.pool_type

	def forward(self, sents, lengths):
		"""
		:param sents: batch x seq_len x 2*d_h1
		:param lengths: numpy array 1 x batch
		:return: batch x d_h1
		"""
		if len(sents.size()) < 2:
			sents = sents.unsqueeze(0)

		utt_embed = self.embeddings(sents)
		sa_mask = get_attn_pad_mask(sents, sents)
		
		# s_embed = self.uttenc(w_embed, lens, sa_mask)
		# s_embed = self.dropout_in(s_embed)  # batch x d_h1
		combined = self.encoder(utt_embed, lengths)

		output1  = self.output1(combined)
		output1  = self.dropout_mid(output1)

		if self.pool_type =='max':
			output1  = torch.max(output1, dim=1)[0]
		elif self.pool_type == 'last':
			output1  = output1[:,-1,:]
		elif self.pool_type == 'mean':
			output1  = torch.mean(output1, dim=1)[0]

		return output1

class TextEncoderOnly(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, args, bert_type='base', encoder_type='GRU', trainable= False):
		super(TextEncoderOnly, self).__init__()

		
		from transformers import BertModel
		if bert_type =='base':
			self.bert = BertModel.from_pretrained('bert-base-uncased')
			self.bert_emb_dim=768
		else:
			self.bert = BertModel.from_pretrained('bert-large-uncased')
			self.bert_emb_dim=768

		for p in self.bert.parameters():
			p.requires_grad = trainable
			
		self.uttenc  = UttEncoder(self.bert_emb_dim, d_h1, args,encoder_type= encoder_type, bidirectional=True)

		if encoder_type =='GRU':
			self.contenc = nn.GRU(d_h1, d_h2, num_layers=1, bidirectional=False)
		else:
			self.contenc = nn.LSTM(d_h1, d_h2, num_layers=1, bidirectional=False)

		self.output1 = nn.Sequential(
			nn.Linear(d_h2, d_h2),
			nn.Tanh()
		)

		self.dropout_small = nn.Dropout(0.1)
		self.dropout_mid   = nn.Dropout(0.5)
		self.pool_type     = args.pool_type


	def forward(self, sents, lens):
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
		s_embed = self.dropout_small(s_embed)  # batch x d_h

		s_context = self.contenc(s_embed.unsqueeze(1))[0]
		combined = s_context.transpose(0,1).contiguous()
		
		output1  = self.output1(combined)
		output1  = self.dropout_mid(output1)

		if self.pool_type =='max':
			output1  = torch.max(output1, dim=1)[0]
		elif self.pool_type == 'last':
			output1  = output1[:,-1,:]
		elif self.pool_type == 'mean':
			output1  = torch.mean(output1, dim=1)[0]

		return output1
		

class TextSeqEncoderSuper(nn.Module):
	def __init__(self, d_word_vec, d_h1, d_h2, d_tag, args, tag2idx, bert_type='base', encoder_type='GRU', trainable= False):
		super(TextSeqEncoderSuper, self).__init__()

		self.text_encoder  =  TextEncoderOnly(d_word_vec, d_h1, d_h2, args)
		self.tag_encoder0  =  SeqEncoderOnly(d_tag,tag2idx,args)
		self.num_tags      =  1
		self.text_encoder1 =  None
		self.tag_encoder1  =  None

		if 'negotiation' in args.dataset:
			self.tag_encoder1  =  SeqEncoderOnly(d_tag,tag2idx,args)
			self.num_tags =  2

		if args.encoder == 'text_seq_encoder':
			self.input_dim  = d_h2+d_tag*self.num_tags
		elif args.encoder == 'text_encoder':
			self.input_dim  = d_h2
		elif args.encoder == 'seq_encoder':
			self.input_dim  = d_tag*self.num_tags

		self.num_classes=2
		self.classifier = nn.Sequential(
			nn.Linear(self.input_dim, self.input_dim),
			nn.Dropout(0.5),
			nn.Linear(self.input_dim, self.num_classes)
		)

		self.dropout_mid     = nn.Dropout(0.5)
		self.dropout_small   = nn.Dropout(0.1)
		self.pool_type = args.pool_type
		self.encoder   = args.encoder
		self.dataset   = args.dataset


	def forward(self, sents, lens, tags, taglens,  tags2= None, taglens2= None):
		"""
		:param sents: batch x seq_len
		:param lens: 1 x batch
		:return:
		"""

		if self.encoder == 'text_seq_encoder':
			sent_output0 =  self.text_encoder(sents, lens)
			tag_output0  =  self.tag_encoder0(tags, taglens)

			if 'negotiation' in self.dataset:
				tag_output1 = self.tag_encoder1(tags2, taglens2)
				combined    = torch.cat([sent_output0, tag_output0, tag_output1], dim =1)
			else:
				combined    = torch.cat([sent_output0, tag_output0], dim =1)

		elif self.encoder == 'text_encoder':
			sent_output0 =  self.text_encoder(sents, lens)
			combined     =  sent_output0

		elif self.encoder == 'seq_encoder':
			tag_output0  =  self.tag_encoder0(tags, taglens)
			
			if 'negotiation' in self.dataset:
				tag_output1 = self.tag_encoder1(tags2, taglens2)
				combined    = torch.cat([tag_output0, tag_output1], dim =1)
			else:
				combined    = tag_output0


		
		combined = self.dropout_small(combined)
		output   = self.classifier(combined)
		pred_scores = F.log_softmax(output, dim=1)

		return pred_scores
		

class CTransformer(nn.Module):
	"""
	Transformer for classifying sequences
	"""

	def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, args, dropout=0.0, wide=False):
		"""
		:param emb: Embedding dimension
		:param heads: nr. of attention heads
		:param depth: Number of transformer blocks
		:param seq_length: Expected maximum sequence length
		:param num_tokens: Number of tokens (usually words) in the vocabulary
		:param num_classes: Number of classes.
		:param max_pool: If true, use global max pooling in the last layer. If false, use global
						 average pooling.
		"""
		super().__init__()

		self.num_tokens = num_tokens

		self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
		self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

		tblocks = []
		for i in range(depth):
			tblocks.append(
				TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout, wide=wide))

		self.tblocks = nn.Sequential(*tblocks)

		self.pool_type = args.pool_type

		self.toprobs = nn.Linear(emb, num_classes)

		self.do = nn.Dropout(dropout)

	def forward(self, x):
		"""
		:param x: A batch by sequence length integer tensor of token indices.
		:return: predicted log-probability vectors for each token based on the preceding tokens.
		"""
		tokens = self.token_embedding(x)
		if len(tokens.size()) <=2:
			tokens = tokens.unsqueeze(0)

		b, t, e = tokens.size()

		positions = self.pos_embedding(torch.arange(t, device=x.device))[None, :, :].expand(b, t, e)
		x = tokens + positions
		x = self.do(x)

		x = self.tblocks(x)

		x = x.max(dim=1)[0] if self.pool_type=='max' else x.mean(dim=1) # pool over the time dimension

		x = self.toprobs(x)

		return F.log_softmax(x, dim=1)





