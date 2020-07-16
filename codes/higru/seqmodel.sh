#!bin/bash

LR2=1e-3
LR=2.5e-3
LR3=2.5e-2
GPU=3
du_bert=768
du_glove=300
dc=300
seed=11747
epochs_num=100


for seqratio in 0.25 0.5 0.75 0.9
do
	python seqmodel.py -lr $LR2 -gpu 0 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_seq_encoder&
	python seqmodel.py -lr $LR2 -gpu 1 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_seq_encoder& 
	python seqmodel.py -lr $LR2 -gpu 2 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_seq_encoder&
	python seqmodel.py -lr $LR2 -gpu 3 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_seq_encoder&
	python seqmodel.py -lr $LR2 -gpu 4 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_seq_encoder&


	python seqmodel.py -lr $LR2 -gpu 5 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_seq_encoder&
	python seqmodel.py -lr $LR2 -gpu 0 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_seq_encoder& 
	python seqmodel.py -lr $LR2 -gpu 1 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_seq_encoder&
	python seqmodel.py -lr $LR2 -gpu 2 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_seq_encoder&
	python seqmodel.py -lr $LR2 -gpu 3 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_seq_encoder&


	python seqmodel.py -lr $LR2 -gpu 4 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_encoder&
	python seqmodel.py -lr $LR2 -gpu 5 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_encoder&	 
	python seqmodel.py -lr $LR2 -gpu 0 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_encoder&	
	python seqmodel.py -lr $LR2 -gpu 1 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_encoder&	
	python seqmodel.py -lr $LR2 -gpu 2 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder text_encoder&	
	

	python seqmodel.py -lr $LR2 -gpu 3 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_encoder&
	python seqmodel.py -lr $LR2 -gpu 4 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_encoder&	 
	python seqmodel.py -lr $LR2 -gpu 5 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_encoder&	
	python seqmodel.py -lr $LR2 -gpu 0 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_encoder&	
	python seqmodel.py -lr $LR2 -gpu 1 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder text_encoder&	
	


	# python seqmodel.py -lr $LR -gpu 0 -count 0 -model_type transformer -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 1 -model_type transformer -seq_ratio $seqratio -pool_type max -encoder seq_encoder& 
	# python seqmodel.py -lr $LR -gpu 0 -count 2 -model_type transformer -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 3 -model_type transformer -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 4 -model_type transformer -seq_ratio $seqratio -pool_type max -encoder seq_encoder&

	# python seqmodel.py -lr $LR -gpu 1 -count 0 -model_type transformer -seq_ratio $seqratio -pool_type mean -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 1 -model_type transformer -seq_ratio $seqratio -pool_type mean -encoder seq_encoder& 
	# python seqmodel.py -lr $LR -gpu 1 -count 2 -model_type transformer -seq_ratio $seqratio -pool_type mean -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 3 -model_type transformer -seq_ratio $seqratio -pool_type mean -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 4 -model_type transformer -seq_ratio $seqratio -pool_type mean -encoder seq_encoder&


	python seqmodel.py -lr $LR -gpu 2 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 3 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 4 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 5 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 0 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type max -encoder seq_encoder&

	python seqmodel.py -lr $LR -gpu 1 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 2 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 3 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 4 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	python seqmodel.py -lr $LR -gpu 5 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type last -encoder seq_encoder&


	# python seqmodel.py -lr $LR -gpu 0 -count 0 -model_type LSTM -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 1 -model_type LSTM -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 2 -model_type LSTM -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 3 -model_type LSTM -seq_ratio $seqratio -pool_type max -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 0 -count 4 -model_type LSTM -seq_ratio $seqratio -pool_type max -encoder seq_encoder&

	# python seqmodel.py -lr $LR -gpu 1 -count 0 -model_type LSTM -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 1 -model_type LSTM -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 2 -model_type LSTM -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 3 -model_type LSTM -seq_ratio $seqratio -pool_type last -encoder seq_encoder&
	# python seqmodel.py -lr $LR -gpu 1 -count 4 -model_type LSTM -seq_ratio $seqratio -pool_type last -encoder seq_encoder&

	wait
done

