LR=1e-3
LR2=1e-4
LR3=5e-5
GPU=3
du_bert=768
du_glove=300
dc=300
seed=11747
epochs_num=100


# python CrossDomain.py -lr  $LR2  -type bert-higru-f -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -data_path ../../data/higru_bert_data/resisting0_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting0_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting0_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting0_tr_coarse_labels_dict_bert.pt -dataset resisting0 -seed $seed -embedding ../../data/higru_bert_data/resisting0_embedding.pt -bert 1 -addn_features $feature

# python CrossDomain.py -lr  $LR  -type higru-f -d_h1 $du_glove -d_h2 $dc -epochs 50 -report_loss 720 -data_path ../../data/higru_bert_data/resisting0_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting0_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting0_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting0_tr_coarse_labels_dict_bert.pt -dataset resisting0 -seed $seed -embedding ../../data/higru_bert_data/resisting0_embedding.pt -bert 0 &

# bert-higru-f bert-higru-sf bert-higru bert-bigru-f bert-bigru bert-bigru-sf


# for model in bert-higru-f bert-higru-sf bert-higru
# do 
# 	for feature in 

# feature_dim_dict = {'vad_features': 3, 'affect_features': 4, 'emo_features': 10, 'liwc_features': 64, 'sentiment_features': 3, 'face_features': 8, 'norm_er_strategies': 10, 'norm_er_DAs': 17, 'ee_DAs': 23, '$feature': 3+4+10+64+3+8+10+17+23}

# vad_features emo_features liwc_features face_features norm_er_strategies ee_DAs norm_er_DAs sentiment_features
model=bert-higru-f
model2=bert-higru-sf

# for feature in emo_features liwc_features sentiment_features none all
for feature in none
do

	python3 CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting0_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting0_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting0_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting0_tr_resistance_labels_dict_bert.pt -dataset resisting0 -seed $seed -embedding ../../data/higru_bert_data/resisting0_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting1_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting1_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting1_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting1_tr_resistance_labels_dict_bert.pt -dataset resisting1 -seed $seed -embedding ../../data/higru_bert_data/resisting1_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting2_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting2_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting2_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting2_tr_resistance_labels_dict_bert.pt -dataset resisting2 -seed $seed -embedding ../../data/higru_bert_data/resisting2_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting3_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting3_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting3_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting3_tr_resistance_labels_dict_bert.pt -dataset resisting3 -seed $seed -embedding ../../data/higru_bert_data/resisting3_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting4_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting4_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting4_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting4_tr_resistance_labels_dict_bert.pt -dataset resisting4 -seed $seed -embedding ../../data/higru_bert_data/resisting4_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature

	# python3 CrossDomain.py -lr  $LR2  -type $model2 -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting0_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting0_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting0_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting0_tr_resistance_labels_dict_bert.pt -dataset resisting0 -seed $seed -embedding ../../data/higru_bert_data/resisting0_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model2 -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting1_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting1_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting1_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting1_tr_resistance_labels_dict_bert.pt -dataset resisting1 -seed $seed -embedding ../../data/higru_bert_data/resisting1_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model2 -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting2_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting2_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting2_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting2_tr_resistance_labels_dict_bert.pt -dataset resisting2 -seed $seed -embedding ../../data/higru_bert_data/resisting2_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model2 -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting3_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting3_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting3_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting3_tr_resistance_labels_dict_bert.pt -dataset resisting3 -seed $seed -embedding ../../data/higru_bert_data/resisting3_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature
	# python3 CrossDomain.py -lr  $LR2  -type $model2 -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting4_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting4_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting4_resistance_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting4_tr_resistance_labels_dict_bert.pt -dataset resisting4 -seed $seed -embedding ../../data/higru_bert_data/resisting4_embedding.pt -bert 1 -label_type resistance -bert_train 0 -addn_features $feature


	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting0_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting0_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting0_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting0_tr_coarse_labels_dict_bert.pt -dataset resisting0 -seed $seed -embedding ../../data/higru_bert_data/resisting0_embedding.pt -bert 1 -label_type coarse -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting1_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting1_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting1_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting1_tr_coarse_labels_dict_bert.pt -dataset resisting1 -seed $seed -embedding ../../data/higru_bert_data/resisting1_embedding.pt -bert 1 -label_type coarse -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting2_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting2_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting2_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting2_tr_coarse_labels_dict_bert.pt -dataset resisting2 -seed $seed -embedding ../../data/higru_bert_data/resisting2_embedding.pt -bert 1 -label_type coarse -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting3_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting3_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting3_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting3_tr_coarse_labels_dict_bert.pt -dataset resisting3 -seed $seed -embedding ../../data/higru_bert_data/resisting3_embedding.pt -bert 1 -label_type coarse -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting4_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting4_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting4_coarse_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting4_tr_coarse_labels_dict_bert.pt -dataset resisting4 -seed $seed -embedding ../../data/higru_bert_data/resisting4_embedding.pt -bert 1 -label_type coarse -bert_train 0 -addn_features $feature&

	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting0_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting0_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting0_fine_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting0_tr_fine_labels_dict_bert.pt -dataset resisting0 -seed $seed -embedding ../../data/higru_bert_data/resisting0_embedding.pt -bert 1 -label_type fine -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting1_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting1_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting1_fine_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting1_tr_fine_labels_dict_bert.pt -dataset resisting1 -seed $seed -embedding ../../data/higru_bert_data/resisting1_embedding.pt -bert 1 -label_type fine -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting2_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting2_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting2_fine_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting2_tr_fine_labels_dict_bert.pt -dataset resisting2 -seed $seed -embedding ../../data/higru_bert_data/resisting2_embedding.pt -bert 1 -label_type fine -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting3_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting3_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting3_fine_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting3_tr_fine_labels_dict_bert.pt -dataset resisting3 -seed $seed -embedding ../../data/higru_bert_data/resisting3_embedding.pt -bert 1 -label_type fine -bert_train 0 -addn_features $feature&
	# python CrossDomain.py -lr  $LR2  -type $model -d_h1 $du_bert -d_h2 $dc -epochs $epochs_num -report_loss 720 -data_path ../../data/higru_bert_data/resisting4_bert_data.pt -vocab_path ../../data/higru_bert_data/resisting4_vocab_bert.pt -emodict_path ../../data/higru_bert_data/resisting4_fine_labels_dict_bert.pt -tr_emodict_path ../../data/higru_bert_data/resisting4_tr_fine_labels_dict_bert.pt -dataset resisting4 -seed $seed -embedding ../../data/higru_bert_data/resisting4_embedding.pt -bert 1 -label_type fine -bert_train 0 -addn_features $feature&
	wait
done

