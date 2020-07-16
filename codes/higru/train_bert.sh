#!bin/bash
LR=1e-3
LR2=2e-5
GPU=3
du_bert=768
# du_bert=300
dc=300

####  NON TRAINABLE BERT    #####



# python EmoMain.py -lr  $LR -gpu 1 -type bert-bigru-sf  -bert_train 0 -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -data_path data/Persuasion/Persuasion0_bert_data.pt -vocab_path data/Persuasion/Persuasion0_vocab_bert.pt -emodict_path data/Persuasion/Persuasion0_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion0_tr_emodict_bert.pt -dataset Persuasion0 -mask 'ER' -ldm 1 -don_model 1 -thresh_reg 0.5 -interpret no_loss -sec_loss mse -alpha 0.9&
# python EmoMain.py -lr  $LR -gpu 2 -type bert-bigru-sf  -bert_train 0 -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -data_path data/Persuasion/Persuasion1_bert_data.pt -vocab_path data/Persuasion/Persuasion1_vocab_bert.pt -emodict_path data/Persuasion/Persuasion1_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion1_tr_emodict_bert.pt -dataset Persuasion1 -mask 'ER' -ldm 1 -don_model 1 -thresh_reg 0.5 -interpret no_loss -sec_loss mse -alpha 0.9&
# python EmoMain.py -lr  $LR -gpu 3 -type bert-bigru-sf  -bert_train 0 -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -data_path data/Persuasion/Persuasion2_bert_data.pt -vocab_path data/Persuasion/Persuasion2_vocab_bert.pt -emodict_path data/Persuasion/Persuasion2_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion2_tr_emodict_bert.pt -dataset Persuasion2 -mask 'ER' -ldm 1 -don_model 1 -thresh_reg 0.5 -interpret no_loss -sec_loss mse -alpha 0.9&
# python EmoMain.py -lr  $LR -gpu 4 -type bert-bigru-sf  -bert_train 0 -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -data_path data/Persuasion/Persuasion3_bert_data.pt -vocab_path data/Persuasion/Persuasion3_vocab_bert.pt -emodict_path data/Persuasion/Persuasion3_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion3_tr_emodict_bert.pt -dataset Persuasion3 -mask 'ER' -ldm 1 -don_model 1 -thresh_reg 0.5 -interpret no_loss -sec_loss mse -alpha 0.9&
# python EmoMain.py -lr  $LR -gpu 0 -type bert-bigru-sf  -bert_train 0 -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -data_path data/Persuasion/Persuasion4_bert_data.pt -vocab_path data/Persuasion/Persuasion4_vocab_bert.pt -emodict_path data/Persuasion/Persuasion4_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion4_tr_emodict_bert.pt -dataset Persuasion4 -mask 'ER' -ldm 1 -don_model 1 -thresh_reg 0.5 -interpret no_loss -sec_loss mse -alpha 0.9&


for seed in 100 11747 42
do 
	for model in bert-higru-sf bert-higru-f bert-higru bert-bigru-sf bert-bigru-f bert-bigru 
	do
		python EmoMain.py -lr  $LR -gpu 0 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion0_bert_data.pt -vocab_path data/Persuasion/Persuasion0_vocab_bert.pt -emodict_path data/Persuasion/Persuasion0_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion0_tr_emodict_bert.pt -dataset Persuasion0 -mask 'EE' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 1 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion1_bert_data.pt -vocab_path data/Persuasion/Persuasion1_vocab_bert.pt -emodict_path data/Persuasion/Persuasion1_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion1_tr_emodict_bert.pt -dataset Persuasion1 -mask 'EE' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 2 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion2_bert_data.pt -vocab_path data/Persuasion/Persuasion2_vocab_bert.pt -emodict_path data/Persuasion/Persuasion2_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion2_tr_emodict_bert.pt -dataset Persuasion2 -mask 'EE' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 3 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion3_bert_data.pt -vocab_path data/Persuasion/Persuasion3_vocab_bert.pt -emodict_path data/Persuasion/Persuasion3_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion3_tr_emodict_bert.pt -dataset Persuasion3 -mask 'EE' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 4 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion4_bert_data.pt -vocab_path data/Persuasion/Persuasion4_vocab_bert.pt -emodict_path data/Persuasion/Persuasion4_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion4_tr_emodict_bert.pt -dataset Persuasion4 -mask 'EE' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &


		python EmoMain.py -lr  $LR -gpu 5 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion0_bert_data.pt -vocab_path data/Persuasion/Persuasion0_vocab_bert.pt -emodict_path data/Persuasion/Persuasion0_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion0_tr_emodict_bert.pt -dataset Persuasion0 -mask 'ER' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 0 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion1_bert_data.pt -vocab_path data/Persuasion/Persuasion1_vocab_bert.pt -emodict_path data/Persuasion/Persuasion1_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion1_tr_emodict_bert.pt -dataset Persuasion1 -mask 'ER' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 1 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion2_bert_data.pt -vocab_path data/Persuasion/Persuasion2_vocab_bert.pt -emodict_path data/Persuasion/Persuasion2_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion2_tr_emodict_bert.pt -dataset Persuasion2 -mask 'ER' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 2 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion3_bert_data.pt -vocab_path data/Persuasion/Persuasion3_vocab_bert.pt -emodict_path data/Persuasion/Persuasion3_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion3_tr_emodict_bert.pt -dataset Persuasion3 -mask 'ER' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 3 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion4_bert_data.pt -vocab_path data/Persuasion/Persuasion4_vocab_bert.pt -emodict_path data/Persuasion/Persuasion4_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion4_tr_emodict_bert.pt -dataset Persuasion4 -mask 'ER' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &


		python EmoMain.py -lr  $LR -gpu 4 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion0_bert_data.pt -vocab_path data/Persuasion/Persuasion0_vocab_bert.pt -emodict_path data/Persuasion/Persuasion0_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion0_tr_emodict_bert.pt -dataset Persuasion0 -mask 'all' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 5 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion1_bert_data.pt -vocab_path data/Persuasion/Persuasion1_vocab_bert.pt -emodict_path data/Persuasion/Persuasion1_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion1_tr_emodict_bert.pt -dataset Persuasion1 -mask 'all' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 3 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion2_bert_data.pt -vocab_path data/Persuasion/Persuasion2_vocab_bert.pt -emodict_path data/Persuasion/Persuasion2_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion2_tr_emodict_bert.pt -dataset Persuasion2 -mask 'all' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 4 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion3_bert_data.pt -vocab_path data/Persuasion/Persuasion3_vocab_bert.pt -emodict_path data/Persuasion/Persuasion3_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion3_tr_emodict_bert.pt -dataset Persuasion3 -mask 'all' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		python EmoMain.py -lr  $LR -gpu 5 -type $model -d_h1 $du_bert -d_h2 $dc -epochs 50 -report_loss 720 -interpret no_loss -data_path data/Persuasion/Persuasion4_bert_data.pt -vocab_path data/Persuasion/Persuasion4_vocab_bert.pt -emodict_path data/Persuasion/Persuasion4_emodict_bert.pt -tr_emodict_path data/Persuasion/Persuasion4_tr_emodict_bert.pt -dataset Persuasion4 -mask 'all' -ldm 1 -don_model 4  -thresh_reg 0.5 -sec_loss mse -alpha 0.9 -seed $seed &
		wait
	done
done
