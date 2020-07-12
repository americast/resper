import sys; sys.path.append('../');
from helper import *
from seq_Modules import *
import Utils
data_dir ='../../data/higru_bert_data/'
from sklearn.metrics import classification_report, accuracy_score, f1_score

'''
Arranged in the form of list of conversations with each 
conversation being a list of turns, with each turn having 
the specific characteristics.
'''

def seed_everything(seed=100):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def return_file_path(args):
	d=vars(args)

	not_d=['patience','d_fc', 'd_h1', 'd_h2', 'd_word_vec','data_path','vocab_path','save_dir', \
	'emodict_path','tr_emodict_path','patience','report_loss','decay','embedding','gpu','max_seq_len', 'bert']

	file_str= ''
	for elem in sorted(d):
		if elem in not_d:
			continue
		
		file_str= file_str+'_'+str(elem)+'_'+str(d[elem])

	return file_str+'_seqmodel'

def get_seq_data(train_data, test_data, ratio):

	tag2idx    = {}
	dataloader = {}
	dataloader['train']  = {}
	dataloader['test']   = {}

	dataloader['train']['tags']    = []
	# dataloader['train']['speaker'] = []
	dataloader['train']['labels']  = []
	dataloader['test']['tags']     = []
	# dataloader['test']['speaker']  = []
	dataloader['test']['labels']   = []

	for conv in train_data:
		for utt in conv:
			if utt['resistance_labels'] not in tag2idx:
				tag2idx[utt['resistance_labels']]= len(tag2idx)

	for conv in train_data:
		tags     = []
		speakers = []
		for i, utt in enumerate(conv):
			if utt['speaker']==0:
				continue
			tags.append(tag2idx[utt['resistance_labels']])
			if i/len(conv)>ratio:
				break
			# speakers.append(utt['speaker'])
		dataloader['train']['tags'].append(tags)			
		# dataloader['train']['speaker'].append(speakers)			
		dataloader['train']['labels'].append([utt['donor']])
			

	for conv in test_data:
		tags     = []
		speakers = []
		for i, utt in enumerate(conv):
			if utt['speaker']==0:
				continue
			tags.append(tag2idx[utt['resistance_labels']])
			if i/len(conv) > ratio:
				break

			# speakers.append(utt['speaker'])
		dataloader['test']['tags'].append(tags)			
		# dataloader['test']['speaker'].append(speakers)			
		dataloader['test']['labels'].append([utt['donor']])

	return dataloader, tag2idx


# for tagseq, label in zip(dataloader['train']['tags'], dataloader['train']['labels']):


def test_model(model, dataloader, args):
	model.eval()
	y_true = []
	y_pred = []
	for bz in range(len(dataloader['tags'])):
		try:
			tagseq = dataloader['tags'][bz]
			label  = dataloader['labels'][bz]

			feat  = Utils.ToTensor(tagseq)
			lens  = np.array([len(tagseq)])
			label = Utils.ToTensor(label)		
			feat  = Variable(feat)
			label  = Variable(label)


			if args.gpu != None:
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				model.cuda(device)
				feat = feat.cuda(device)
				label = label.cuda(device)

			if args.model_type =='transformer':
				log_prob = model(feat)
			else:
				log_prob = model(feat, lens)
			
			don_true = label.view(label.size(0))

			don_true = np.array(don_true.cpu())

			don_predidx = torch.argmax(log_prob, dim=1)
			don_predidx= np.array(don_predidx.cpu())

			y_pred.extend(don_predidx)
			y_true.extend(don_true)
		except Exception as e:
			import pdb; pdb.set_trace()

	f1  = f1_score(y_true, y_pred, average='macro')
	acc = accuracy_score(y_true, y_pred)

	

	return f1, acc

def train_model(model, dataloader, args):

	lr = args.lr
	model_opt = optim.Adam(model.parameters(), lr=lr)
	model.train()

	decay = args.decay
	over_fitting = 0
	curr_best = -1e10
	glob_steps = 0
	report_loss = 0
	eps = 1e-10
	st = time.time()

	file_str = return_file_path(args)
	f=open('../../data/higru_bert_data/results_seqmodel/'+file_str+ '.txt','w')
	for epoch in range(1, args.epochs + 1):
		model_opt.param_groups[0]['lr'] *= decay	# Decay the lr every epoch

		print("===========Epoch==============")
		print("-{}-{}".format(epoch, time.time()-st))

		for bz in range(len(dataloader['train']['tags'])):

			tagseq = dataloader['train']['tags'][bz]
			label  = dataloader['train']['labels'][bz]

			feat  = Utils.ToTensor(tagseq)
			lens  = np.array([len(tagseq)])
			label = Utils.ToTensor(label)		
			feat  = Variable(feat)
			label  = Variable(label)


			if args.gpu != None:
				os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
				device = torch.device("cuda: 0")
				model.cuda(device)
				feat = feat.cuda(device)
				label = label.cuda(device)


			if args.model_type =='transformer':
				log_prob = model(feat)
			else:
				log_prob = model(feat, lens)


			target   = label

			nll_loss = nn.NLLLoss()
			loss= nll_loss(log_prob, target)

			if loss !=loss:
				import pdb; pdb.set_trace()

			loss.backward()

			report_loss += loss.item()
			glob_steps += 1

			# gradient clip
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

			model_opt.step()
			model_opt.zero_grad()
		
			if glob_steps % 100 == 0:
				print("Steps: {} Loss: {} LR: {}".format(glob_steps, report_loss/10, model_opt.param_groups[0]['lr']))
				report_loss = 0
		
		mf1, acc = test_model(model=model, dataloader=dataloader['train'],args=args)

		print("Train acc = {}".format(acc))
		print("Train F1 = {}".format(mf1))
		
		mf1, acc = test_model(model=model, dataloader= dataloader['test'],args=args)

		print("Test acc = {}".format(acc))
		print("Test F1 = {}".format(mf1))
		
		f.write(str(epoch)+'\t'+str(acc)+'\t'+str(mf1)+'\n')
		model.train()

		# if mf1 > curr_best:
		# 	torch.save(model, args.save_dir+file_str+'_model.pt')
		# 	curr_best = mf1
		# 	over_fitting = 0
		# else:
		# 	over_fitting += 1


def main():


	parser = argparse.ArgumentParser()

	parser.add_argument('-lr', type=float, default=2.5e-3)		# Learning rate: 2.5e-4 for Friends and EmotionPush, 1e-4 for IEMOCAP
	parser.add_argument('-decay', type=float, default=math.pow(0.5, 1/20))	# half lr every 20 epochs
	parser.add_argument('-epochs', type=int, default=100)		# Defualt epochs 200
	# parser.add_argument('-patience', type=int, default=10,help='patience for early stopping') 
	parser.add_argument('-gpu', type=str, default='0')
	parser.add_argument('-count', type= str, default='0')
	parser.add_argument('-model_type', type= str, default='GRU')
	parser.add_argument('-save_dir', type=str, default="/data/politeness_datasets/snapshot_models")	# Save the model and results in 
	parser.add_argument('-pool_type', type=str, default='max')
	parser.add_argument('-seed', type = int, default=100)
	parser.add_argument('-seq_ratio',type= float, default=1)

	args = parser.parse_args()

	seed_everything(args.seed)

	train_data = json.load(open(data_dir+'train'+args.count+'.json'))
	test_data  = json.load(open(data_dir+'test'+args.count+'.json'))

	max_train_len= max([len(i) for i in train_data])
	max_test_len = max([len(i) for i in test_data])
	max_len = int(max(max_test_len, max_train_len)*args.seq_ratio)+1

	dataloader, tag2idx = get_seq_data(train_data, test_data, args.seq_ratio)

	print(len(dataloader['train']['tags']))
	print(len(dataloader['test']['tags']))
	print(tag2idx)

	if args.model_type =='transformer':
		model = CTransformer(emb=10, heads=1, depth=2, seq_length=max_len, num_tokens=len(tag2idx), num_classes=2, args=args, dropout=0.5, wide=False)
	
	else:
		model = SeqEncoder(10,tag2idx,args)

	train_model(model, dataloader, args)


main()