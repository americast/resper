"""
Train on Emotion dataset
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import Utils
import math
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import pdb
from sklearn.utils.class_weight import compute_sample_weight


def return_addn_features(data_loader, args):

	feat_names = ['vad_features', 'affect_features', 'emo_features', 'liwc_features', 'sentiment_features', 'face_features', 'norm_er_strategies', 'norm_er_DAs', 'ee_DAs']

	if args.addn_features in feat_names:
		return data_loader[args.addn_features]

	addn_features_arr = None

	if args.addn_features =='all':
		conv_nums = len(data_loader['feat'])

		addn_features_arr = []
		for conv_num in range(conv_nums):
			addn_features = None
			for index, feat_name in enumerate(feat_names):
				feat_data = data_loader[feat_name][conv_num]
				if index == 0:
					addn_features = np.array(feat_data)
				else:
					addn_features = np.hstack((addn_features, np.array(feat_data)))

			addn_features_arr.append(addn_features)

	return addn_features_arr

def emotrain(model, data_loader, tr_emodict, emodict, args, focus_emo):
	"""
	:data_loader input the whole field
	"""
	# start time
	time_st = time.time()
	decay_rate = args.decay
	alpha= 1.0

	# Load in the training set and validation set
	train_loader   =   data_loader['train']
	dev_loader     =   data_loader['test']


	if args.bert == 1:
		feats = train_loader['bert-feat']
	else:
		feats = train_loader['feat']

	labels  =   train_loader[args.label_type+'_labels']

	speakers       =   train_loader['speaker']
	addn_features  =   return_addn_features(train_loader, args)

	# Optimizer
	lr = args.lr
	model_opt = optim.Adam(model.parameters(), lr=lr)

	# Weight for loss

	'''
	Removing the auxillary weights condition. 
	'''

	# Raise the .train() flag before training
	model.train()

	# criterion = torch.nn.BCELoss(pos_weights=torch.Tensor([0.2]), reduction='none')

	file_str = Utils.return_file_path(args)
	

	f=open('../../data/higru_bert_data/results/'+file_str+ '.txt','w')
	

	over_fitting = 0
	cur_best = -1e10
	cur_face_best = -1e10
	glob_steps = 0
	report_loss = 0
	eps = 1e-10
	for epoch in range(1, args.epochs + 1):
		model_opt.param_groups[0]['lr'] *= decay_rate	# Decay the lr every epoch

		# import pdb; pdb.set_trace()

		if addn_features != None:
			feats, labels, speakers, addn_features= Utils.shuffle_lists(feats, labels, speakers, addn_features)
		else:
			feats, labels, speakers = Utils.shuffle_lists(feats, labels, speakers)
		
			# sample_weights)	# Shuffle the training set every epoch
		print("===========Epoch==============")
		print("-{}-{}".format(epoch, Utils.timeSince(time_st)))

		for bz in range(len(labels)):
			# Tensorize a dialogue, a dialogue is a batch
			feat, lens = Utils.ToTensor(feats[bz], is_len=True)
			label = Utils.ToTensor(labels[bz])
			if 'negotiation' in args.dataset:
				mask  = torch.LongTensor([1 for i in speakers[bz]])
			else:
				mask  = torch.LongTensor([int(i) for i in speakers[bz]])
			addn_feature = None
			
			# EE_mask= torch.LongTensor([int(i) for i in speakers[bz]])
			# ER_mask = torch.LongTensor([1-int(i) for i in speakers[bz]])

			# EE_weights = torch.FloatTensor([0 if i in ['4'] else 1 for i,j in emodict.word2index.items()])
			# ER_weights = torch.FloatTensor([0 if i in ['0','7'] else 1 for i,j in emodict.word2index.items()])

			# donor_mask= torch.LongTensor([0 for i in range(len(speakers[bz]))])	
			# donor_mask[len(donor_mask)-args.ldm:]=1

			feat = Variable(feat)
			label = Variable(label)
			# bert_emb= Variable(bert_emb)

			if args.gpu != "cpu":
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				model.cuda(device)
				feat = feat.cuda(device)
				label = label.cuda(device)
				mask= mask.cuda(device)


				# donor_mask= donor_mask.cuda(device)
				# donor_label= donor_label.cuda(device)
				# donor_float_label = donor_float_label.cuda(device)
				# EE_mask = EE_mask.cuda(device)
				# ER_mask = ER_mask.cuda(device)
				# ER_weights = ER_weights.cuda(device)
				# EE_weights = EE_weights.cuda(device)
				# weights = weights.cuda(device)

			if addn_features != None:
				addn_feature = torch.FloatTensor(addn_features[bz])
				addn_feature = Variable(addn_feature)
				addn_feature = addn_feature.cuda(device)
			
			


			log_prob,  log_donor_prob, pred_outs = model(feat, lens, addn_feature)
			target   = label
			all_loss = torch.gather(log_prob, 1, target).squeeze(1)				
			# if all_loss !=all_loss:

			# import pdb; pdb.set_trace()

			loss  = -(all_loss*mask).sum()/mask.sum()

			if loss !=loss:
				import pdb; pdb.set_trace()

			loss.backward()

			loss2 = None

			# if log_donor_prob == None:
			# 	if args.sec_loss =='mse':
			# 		mse   = torch.nn.MSELoss(reduction='sum')
			# 		loss2 = mse(pred_outs[-1],donor_float_label[-1])

			# 	else:

			# 		logits = torch.log(pred_outs/(1+eps-pred_outs))
			# 		loss2  = F.binary_cross_entropy_with_logits(logits, donor_float_label,reduction='none')

			# 		# loss2  = F.binary_cross_entropy_with_logits(logits, donor_float_label, pos_weight=torch.Tensor([0.2]).cuda(device),reduction='none')
			# 		loss2  = (loss2.squeeze(1)*donor_mask).sum()
			# 		# loss2 = F.binary_cross_entropy(pred_outs.reshape(-1, 1), donor_float_label, weights=sample_weights[bz])
			# 		# loss2 = criterion(pred_outs.reshape(-1,1), donor_float_label)*donor_mask

			# 	if loss2!=loss2:
			# 		import pdb; pdb.set_trace()
				

			# else:
			# 	loss2 = torch.gather(log_donor_prob, 1, donor_label).squeeze(1)*donor_mask
			# 	loss2 = -loss2.sum()/donor_mask.sum()
	
			# if args.interpret  =='combined_trainable_loss': # add both losses as it happens
			# 	loss = loss + loss2
			# 	loss.backward()
			# elif args.interpret == 'single_loss':  # add them in a weighed fashion
			# 	loss = alpha*loss + (1- alpha)*loss2
			# 	loss.backward()
			# elif args.interpret =='combined_non_trainable_loss':
			# 	loss.backward(retain_graph=True)
			# 	param_list=[]
			# 	for name, param in model.named_parameters():
			# 		if name.startswith('classifier2')==False and param.requires_grad:
			# 			param.requires_grad=False
			# 			param_list.append((name, param))
			# 	loss2.backward()
			# else:
			# 	loss.backward()


			report_loss += loss.item()
			glob_steps += 1

			# gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

			model_opt.step()
			model_opt.zero_grad()

			if glob_steps % args.report_loss == 0:
				print("Steps: {} Loss: {} LR: {}".format(glob_steps, report_loss/args.report_loss, model_opt.param_groups[0]['lr']))
				report_loss = 0
		
		pAccs, acc, mf1 = emoeval(model=model, data_loader=train_loader, tr_emodict=tr_emodict, emodict=emodict, args=args, focus_emo=focus_emo)

		print("Train acc = {}".format(acc))
		print("Train F1 = {}".format(mf1))
		

		# validate
		pAccs, acc, mf1 = emoeval(model=model, data_loader=dev_loader, tr_emodict=tr_emodict, emodict=emodict, args=args, focus_emo=focus_emo)

		# print("Validate: ACCs-WA-UWA {}".format(pAccs))
		print("Validation acc = {}".format(acc))
		print("Validation F1 = {}".format(mf1))
		

		f.write(str(epoch)+'\t'+str(acc)+'\t'+str(mf1)+'\n')

		# last_don_best= don_mf1
		last_best = mf1

		if last_best > cur_face_best:
			# torch.save(model.state_dict(), args.save_dir+'/'+file_str+'.pt')

			torch.save(model, args.save_dir+'/'+file_str+'_model.pt')
			
			# Utils.model_saver(model, args.save_dir, args.type, args.dataset, args)
			cur_face_best = last_best
			over_fitting = 0
		else:
			over_fitting += 1


		if over_fitting > args.patience:
			print("Best performance before breaking acc {} f1 {} ".format(acc, mf1))
			break


def comput_class_loss(log_prob, target, weights):
	""" Weighted loss function """
	loss = F.nll_loss(log_prob, target.view(target.size(0)), weight=weights, reduction='sum')
	loss /= target.size(0)

	return loss


def loss_weight(tr_ladict, ladict, focus_dict, rate=1.0):
	""" Loss weights """
	min_emo = float(min([tr_ladict.word2count[w] for w in focus_dict]))
	weight = [math.pow(min_emo / tr_ladict.word2count[k], rate) if k in focus_dict
	          else 0 for k,v in ladict.word2count.items()]
	weight = np.array(weight)
	weight /= np.sum(weight)

	return weight


def emoeval(model, data_loader, tr_emodict, emodict, args, focus_emo):
	""" data_loader only input 'dev' """
	model.eval()

	if args.bert == True:
		feats = data_loader['bert-feat']
	else:
		feats = data_loader['feat']

	labels  =   data_loader[args.label_type+'_labels']

	# feats, labels = data_loader['feat'], data_loader['label']
	texts         = data_loader['text']
	# bert_embs     = data_loader['bert-feat']
	speakers      = data_loader['speaker']

	# df_dict = {}
	# df_dict['conversation_id'] =  []
	# df_dict['speaker']         =  []
	# df_dict['utterance']       =  []
	# df_dict['predicted_face']  =  []
	# df_dict['true_face']       =  []
	# df_dict['actual_donation'] =  []
	# df_dict['donation_prob']   =  []

	addn_features = return_addn_features(data_loader, args)
	

	alpha= 1.0
	# donors= data_loader['donor']

	# donor_probs=[]

	val_loss = 0
	y_true=[]
	y_pred=[]

	# old_y_true = []
	# old_y_pred = []

	# donor_true=[]
	# donor_pred=[]

	# donor_labels = []
	# donor_logits = []

	# EE_true = []
	# EE_pred = []
	# ER_true = []
	# ER_pred = []

	# pred_face_arr = []
	# true_face_arr = []

	# curr_conv_id = int(args.dataset[-1])*60

	for bz in range(len(labels)):
		if texts!=None:
			turns = texts[bz]

		feat, lens = Utils.ToTensor(feats[bz], is_len=True)
		label = Utils.ToTensor(labels[bz])


		# bert_emb = np.zeros((len(bert_embs[bz]),max([len(b) for b in bert_embs[bz]])+2))
		# bert_emb = torch.FloatTensor(bert_embs[bz])

		# donor_label= torch.LongTensor(donors[bz]).unsqueeze(dim=1)
		
		if 'negotiation' in args.dataset:
			mask  = torch.LongTensor([1 for i in speakers[bz]])
		else:
			mask  = torch.LongTensor([int(i) for i in speakers[bz]])

		addn_feature = None
		# if args.mask=='EE':
		# 	mask= torch.LongTensor([int(i) for i in speakers[bz]])
		# elif args.mask=='ER':
		# 	mask = torch.LongTensor([1-int(i) for i in speakers[bz]])
		# else:
		# 	mask= torch.LongTensor([1 for i in speakers[bz]])

		# EE_mask = np.array([int(i) for i in speakers[bz]])
		# ER_mask = np.array([1-int(i) for i in speakers[bz]])

		# EE_weights = torch.FloatTensor([0 if i in ['4'] else 1 for i,j in emodict.word2index.items()])
		# ER_weights = torch.FloatTensor([0 if i in ['0','7'] else 1 for i,j in emodict.word2index.items()])


		# donor_mask= torch.LongTensor([0 for i in range(len(speakers[bz]))])	
		# donor_mask[len(donor_mask)-args.ldm:]=1

		feat = Variable(feat)
		label = Variable(label)
		# bert_emb= Variable(bert_emb)
		
		if args.gpu != "cpu":
			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
			device = torch.device("cuda: 0")
			model.cuda(device)
			feat = feat.cuda(device)
			label = label.cuda(device)
			# bert_emb= bert_emb.cuda(device)
			mask= mask.cuda(device)
			# donor_label= donor_label.cuda(device)
			# donor_mask= donor_mask.cuda(device)
			# EE_weights= EE_weights.cuda(device)
			# ER_weights= ER_weights.cuda(device)
			# weights = weights.cuda(device)

		if addn_features != None:
			addn_feature = torch.FloatTensor(addn_features[bz])
			addn_feature = Variable(addn_feature)
			addn_feature = addn_feature.cuda(device)

		log_prob, log_donor_prob, pred_outs = model(feat, lens, addn_feature)

		# val loss
		# loss = comput_class_loss(log_prob, label, weights)
		target=label
		emo_true = label.view(label.size(0))
		emo_true = np.array(emo_true.cpu())


		# Avoid computing loss since it is not used for computation here. 

		# face act prediction stuff done here.......

		face_pred = []
		face_true = []

		'''
		if args.mask =='all':
			# old_predidx = torch.argmax(log_prob,)
			#  pdb.set_trace()

			old_predidx = torch.argmax(log_prob, dim =1)
			old_predidx = np.array(old_predidx.cpu())

			face_prob = torch.exp(log_prob)
			EE_face   = face_prob*EE_weights
			ER_face   = face_prob*ER_weights

			# ensures that the unwanted faces are obscured for EE_face and ER_face

			ER_predidx= torch.argmax(ER_face, dim=1)
			EE_predidx= torch.argmax(EE_face, dim=1)

			ER_predidx= np.array(ER_predidx.cpu())
			EE_predidx= np.array(EE_predidx.cpu())

			ER_pred.extend([i for i,j in zip(ER_predidx, ER_mask) if j==1])
			ER_true.extend([i for i,j in zip(emo_true, ER_mask) if j==1])

			EE_pred.extend([i for i,j in zip(EE_predidx, EE_mask) if j==1])
			EE_true.extend([i for i,j in zip(emo_true, EE_mask) if j==1])

			y_pred.extend([i for i,j in zip(ER_predidx, ER_mask) if j==1])
			y_true.extend([i for i,j in zip(emo_true, ER_mask) if j==1])
			y_pred.extend([i for i,j in zip(EE_predidx, EE_mask) if j==1])
			y_true.extend([i for i,j in zip(emo_true, EE_mask) if j==1])

			mask= np.array(mask.cpu())
			old_y_pred.extend([i for i,j in zip(old_predidx, mask) if j==1])
			old_y_true.extend([i for i,j in zip(emo_true, mask) if j==1])


			inv_face_act = {'0':'spos-', '1':'hpos+', '2':'other', '3':'spos+', '4':'hneg+','5':'hpos-','6':'hneg-','7':'sneg+'}

			for i in range(len(emo_true)):
				face_true.append(inv_face_act[emodict.index2word[emo_true[i]]])
				if ER_mask[i] == 1:
					face_pred.append(inv_face_act[emodict.index2word[ER_predidx[i]]])
				else:
					face_pred.append(inv_face_act[emodict.index2word[EE_predidx[i]]])

			pred_face_arr.append(face_pred)
			true_face_arr.append(face_true)

		'''
		

		emo_predidx = torch.argmax(log_prob, dim=1)
		# emo_true = label.view(label.size(0))
		emo_predidx= np.array(emo_predidx.cpu())
		# emo_true= np.array(emo_true.cpu())
		mask= np.array(mask.cpu())
		y_pred.extend([i for i,j in zip(emo_predidx, mask) if j==1])
		y_true.extend([i for i,j in zip(emo_true, mask) if j==1])


		# donor prediction and accuracy computed here. 

		# donor_trueidx= donor_label.view(donor_label.size(0))
		# donor_trueidx= np.array(donor_trueidx.cpu())
		# donor_mask= np.array(donor_mask.cpu())

		# donor_predidx = None
		# dons = None

		# if log_donor_prob !=None:

		# 	dons=np.array(F.softmax(log_donor_prob, dim=1).cpu().detach())
		# 	donor_probs.append(dons)
		# 	donor_predidx= torch.argmax(log_donor_prob, dim=1)
		# 	donor_predidx= np.array(donor_predidx.cpu())
			
			
		# else:
		# 	dons = np.array(pred_outs.cpu().detach())
		# 	donor_probs.append(dons)
		# 	# donor_predidx =[1 if elem[0] >args.thresh_reg else 0 for elem in dons]
		# 	donor_predidx =[1 if elem[0] >0.45 else 0 for elem in dons]



		
		# donor_pred.extend([i for i,j in zip(donor_predidx, donor_mask) if j==1])
		# donor_true.extend([i for i,j in zip(donor_trueidx, donor_mask) if j==1])

		# donor_logits.extend([i for i,j in zip([elem[0] for elem in dons], donor_mask) if j==1])

		# face_labels =  {'spos-': 7, 'hpos+': 1, 'other': 0, 'spos+': 3, 'hneg+': 5, 'hpos-': 4, 'hneg-': 2, 'sneg+': 6}
		# inv_face_labels = {}
		# for face_act in face_labels:
		# 	inv_face_labels[face_labels[face_act]] = face_act


	# 	if texts!=None:
	# 		for turn, emo, emo_true, don, act_don, speaker in zip(turns, face_pred, face_true, dons, donor_trueidx, speakers[bz]):
	# 			df_dict['conversation_id'].append(curr_conv_id)
	# 			df_dict['speaker'].append(speaker)
	# 			df_dict['utterance'].append(turn)
	# 			df_dict['donation_prob'].append(don[-1])
	# 			df_dict['predicted_face'].append(emo)
	# 			df_dict['true_face'].append(emo_true)
	# 			df_dict['actual_donation'].append(act_don)
				

	# 	curr_conv_id +=1

	# df = pd.DataFrame(df_dict)
	# file_str = Utils.return_file_path(args)

	# df.to_csv('/projects/persuasionforgood-master/MIT-projects/results/'+file_str+ '.csv')
	# # df.to_csv('result_csv/'+str(file_str)+'.csv')

	model.train()
	Total=val_loss

	acc=accuracy_score(y_true,y_pred)
	mf1= f1_score(y_true,y_pred,average='macro')

	# don_acc= accuracy_score(donor_true, donor_pred)
	# don_mf1= f1_score(donor_true, donor_pred, average='macro')

	# t, max_f1 = tune_thresholds(np.array(donor_true).reshape(-1,1), np.array(donor_logits).reshape(-1,1))
	
	# new_donor_pred = [1 if elem > t else 0 for elem in donor_logits]

	# new_don_acc= accuracy_score(donor_true, new_donor_pred)
	# new_don_mf1= f1_score(donor_true, new_donor_pred, average='macro')

	# # print(donor_logits)
	# # print(donor_true)


	print(classification_report(y_true, y_pred))

	# if old_mf1 > mf1:
	# 	 pdb.set_trace()

	# print(classification_report(ER_true, ER_pred))
	# print(classification_report(EE_true, EE_pred))

	print(acc, mf1)
	# print(don_acc, don_mf1)
	# print(new_don_mf1, new_don_mf1)
	# print(t)	

	return Total, acc, mf1#, don_acc, don_mf1, donor_probs  


def tune_thresholds(labels, logits, method = 'tune'):
	'''
	Takes labels and logits and tunes the thresholds using two methods
	methods are 'tune' or 'zc' #Zach Lipton
	Returns a list of tuples (thresh, f1) for each feature
	'''
	if method not in ['tune', 'zc']:
		print ('Tune method should be either tune or zc')
		sys.exit(1)

	# def sigmoid(x):
	# 	return 1/1+np.exp(-x)
	
	res = []
	# logits = sigmoid(logits)

	num_labels = labels.shape[1]


	def tune_th(pid, feat):
		max_f1, th = 0, 0		# max f1 and its thresh
		if method == 'tune':
			ts_to_test = np.arange(0, 1, 0.001)
			for t in ts_to_test:
				scr  = f1_score(labels[:, feat], logits[:, feat] > t, average='macro')
				if scr > max_f1:
					max_f1	= scr
					th	= t
		else:
			f1_half = f1_score(labels[:, feat], logits[:, feat] > 0.5, average='macro')
			th = f1_half / 2
			max_f1 = f1_score(labels[:, feat], logits[:, feat] > th, average='macro')

		return (th, max_f1)
		
	res = tune_th(0, 0)
	return res

# def emoeval2(model, data_loader, tr_emodict, emodict, args, focus_emo, texts):
# 	""" data_loader only input 'dev' """
# 	model.eval()

# 	# weight for loss
# 	# weight_rate = 0.75 # eval state without weights
# 	# if args.dataset in ['IEMOCAP']:
# 	# 	weight_rate = 0
# 	# weights = torch.from_numpy(loss_weight(tr_emodict, emodict, focus_emo, rate=weight_rate)).float()

# 	# TP = np.zeros([emodict.n_words], dtype=np.long) # recall
# 	# TP_FN = np.zeros([emodict.n_words], dtype=np.long) # gold
# 	# focus_idx = [emodict.word2index[emo] for emo in focus_emo]

# 	feats, labels = data_loader['feat'], data_loader['label']
# 	bert_embs= data_loader['bert-feat']
# 	speakers= data_loader['speaker']

# 	# donor_data= data_loader['donor']

# 	alpha= args.alpha
# 	donors= data_loader['donor']

# 	donor_probs=[]

# 	val_loss = 0
# 	y_true=[]
# 	y_pred=[]
# 	donor_true=[]
# 	donor_pred=[]

# 	out_arr=[]

# 	for bz in range(len(labels)):
# 		turns = texts[bz]
# 		feat, lens = Utils.ToTensor(feats[bz], is_len=True)
# 		label = Utils.ToTensor(labels[bz])
# 		bert_emb= torch.FloatTensor(bert_embs[bz])
# 		donor_label= torch.LongTensor(donors[bz]).unsqueeze(dim=1)
		


# 		if args.mask=='EE':
# 			mask= torch.LongTensor([int(i) for i in speakers[bz]])
# 		elif args.mask=='ER':
# 			mask = torch.LongTensor([1-int(i) for i in speakers[bz]])
# 		else:
# 			mask= torch.LongTensor([1 for i in speakers[bz]])

# 		donor_mask= torch.LongTensor([0 for i in range(len(speakers[bz]))])	
# 		donor_mask[len(donor_mask)-1]=1

# 		feat = Variable(feat)
# 		label = Variable(label)
# 		bert_emb= Variable(bert_emb)
		
# 		if args.gpu != None:
# 			os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# 			device = torch.device("cuda: 0")
# 			model.cuda(device)
# 			feat = feat.cuda(device)
# 			label = label.cuda(device)
# 			bert_emb= bert_emb.cuda(device)
# 			mask= mask.cuda(device)
# 			donor_label= donor_label.cuda(device)
# 			donor_mask= donor_mask.cuda(device)
			
# 			# weights = weights.cuda(device)

# 		log_prob, log_donor_prob,_ = model(feat, lens, bert_emb)#, speaker_em)

# 		dons=np.array(F.softmax(log_donor_prob, dim=1).cpu().detach())
# 		donor_probs.append(dons)



# 		# val loss
# 		# loss = comput_class_loss(log_prob, label, weights)
# 		target=label

# 		all_loss = torch.gather(log_prob, 1, target).squeeze(1)
# 		loss= -(all_loss*mask).sum()/mask.sum()

# 		'''
# 		Donor loss here : 
# 		'''
# 		loss2= torch.gather(log_donor_prob, 1, donor_label).squeeze(1)* donor_mask
# 		loss2= -loss2.sum()
		
# 		val_loss += loss.item()

# 		# accuracy
# 		emo_predidx = torch.argmax(log_prob, dim=1)
# 		emo_true = label.view(label.size(0))
		
# 		emo_predidx= np.array(emo_predidx.cpu())
# 		emo_true= np.array(emo_true.cpu())
# 		mask= np.array(mask.cpu())

# 		# out_arr.append(np.array(outs.cpu().detach()))

# 		'''
# 		Get the donor values.		

# 		'''
# 		donor_predidx= torch.argmax(log_donor_prob, dim=1)
# 		donor_trueidx= donor_label.view(donor_label.size(0))

# 		donor_predidx= np.array(donor_predidx.cpu())
# 		donor_trueidx= np.array(donor_trueidx.cpu())
# 		donor_mask= np.array(donor_mask.cpu())


# 		y_pred.extend([i for i,j in zip(emo_predidx, mask) if j==1])
# 		y_true.extend([i for i,j in zip(emo_true, mask) if j==1])

# 		donor_pred.extend([i for i,j in zip(donor_predidx, donor_mask) if j==1])
# 		donor_true.extend([i for i,j in zip(donor_trueidx, donor_mask) if j==1])





# 	# 	for lb in range(emo_true.size(0)):
# 	# 		idx = emo_true[lb].item()
# 	# 		TP_FN[idx] += 1
# 	# 		if idx in focus_idx:
# 	# 			if emo_true[lb] == emo_predidx[lb]:
# 	# 				TP[idx] += 1

# 	# f_TP = [TP[emodict.word2index[w]] for w in focus_emo]
# 	# f_TP_FN = [TP_FN[emodict.word2index[w]] for w in focus_emo]
# 	# Recall = [np.round(tp/tp_fn*100, 2) if tp_fn>0 else 0 for tp,tp_fn in zip(f_TP,f_TP_FN)]
# 	# wRecall = sum([r * w / sum(f_TP_FN) for r,w in zip(Recall, f_TP_FN)])
# 	# uRecall = sum(Recall) / len(Recall)

# 	# Accuracy of each class w.r.t. the focus_emo, the weighted acc, and the unweighted acc
# 	# Total = Recall + [np.round(wRecall,2), np.round(uRecall,2)]

# 	# Return to .train() state after validation
# 	model.train()
# 	Total=0

# 	acc=accuracy_score(y_true,y_pred)
# 	mf1= f1_score(y_true,y_pred,average='macro')

# 	don_acc= accuracy_score(donor_true, donor_pred)
# 	don_mf1= f1_score(donor_true, donor_pred, average='macro')


# 	print(acc, mf1)
# 	print(don_acc, don_mf1)
# 	return Total, acc, mf1, don_acc, don_mf1, donor_probs  #, np.array(out_arr), donor_data, speakers
