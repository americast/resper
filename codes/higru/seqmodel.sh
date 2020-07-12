#!bin/bash

LR=2.5e-3
GPU=3
du_bert=768
du_glove=300
dc=300
seed=11747
epochs_num=100


for seqratio in 0.25 0.5 0.75 0.9 1.0
do
	python seqmodel.py -lr $LR -gpu 0 -count 0 -model_type transformer -seq_ratio $seqratio -pool_type max&
	python seqmodel.py -lr $LR -gpu 1 -count 1 -model_type transformer -seq_ratio $seqratio -pool_type max& 
	python seqmodel.py -lr $LR -gpu 3 -count 2 -model_type transformer -seq_ratio $seqratio -pool_type max&
	python seqmodel.py -lr $LR -gpu 4 -count 3 -model_type transformer -seq_ratio $seqratio -pool_type max&
	python seqmodel.py -lr $LR -gpu 5 -count 4 -model_type transformer -seq_ratio $seqratio -pool_type max&

	python seqmodel.py -lr $LR -gpu 0 -count 0 -model_type transformer -seq_ratio $seqratio -pool_type mean&
	python seqmodel.py -lr $LR -gpu 1 -count 1 -model_type transformer -seq_ratio $seqratio -pool_type mean& 
	python seqmodel.py -lr $LR -gpu 3 -count 2 -model_type transformer -seq_ratio $seqratio -pool_type mean&
	python seqmodel.py -lr $LR -gpu 4 -count 3 -model_type transformer -seq_ratio $seqratio -pool_type mean&
	python seqmodel.py -lr $LR -gpu 5 -count 4 -model_type transformer -seq_ratio $seqratio -pool_type mean&


	# python seqmodel.py -lr $LR -gpu 0 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type max&
	# python seqmodel.py -lr $LR -gpu 1 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type max& 
	# python seqmodel.py -lr $LR -gpu 2 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type max&
	# python seqmodel.py -lr $LR -gpu 0 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type max&
	# python seqmodel.py -lr $LR -gpu 1 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type max&

	# python seqmodel.py -lr $LR -gpu 2 -count 0 -model_type GRU -seq_ratio $seqratio -pool_type last&
	# python seqmodel.py -lr $LR -gpu 0 -count 1 -model_type GRU -seq_ratio $seqratio -pool_type last&  
	# python seqmodel.py -lr $LR -gpu 1 -count 2 -model_type GRU -seq_ratio $seqratio -pool_type last&
	# python seqmodel.py -lr $LR -gpu 2 -count 3 -model_type GRU -seq_ratio $seqratio -pool_type last&
	# python seqmodel.py -lr $LR -gpu 3 -count 4 -model_type GRU -seq_ratio $seqratio -pool_type last&


	# python seqmodel.py -lr $LR -gpu 4 -count 0 -model_type LSTM -seq_ratio $seqratio -pool_type max&
	# python seqmodel.py -lr $LR -gpu 4 -count 1 -model_type LSTM -seq_ratio $seqratio -pool_type max& 
	# python seqmodel.py -lr $LR -gpu 4 -count 2 -model_type LSTM -seq_ratio $seqratio -pool_type max&
	# python seqmodel.py -lr $LR -gpu 4 -count 3 -model_type LSTM -seq_ratio $seqratio -pool_type max&
	# python seqmodel.py -lr $LR -gpu 4 -count 4 -model_type LSTM -seq_ratio $seqratio -pool_type max&

	# python seqmodel.py -lr $LR -gpu 5 -count 0 -model_type LSTM -seq_ratio $seqratio -pool_type last&
	# python seqmodel.py -lr $LR -gpu 5 -count 1 -model_type LSTM -seq_ratio $seqratio -pool_type last&  
	# python seqmodel.py -lr $LR -gpu 5 -count 2 -model_type LSTM -seq_ratio $seqratio -pool_type last&
	# python seqmodel.py -lr $LR -gpu 5 -count 3 -model_type LSTM -seq_ratio $seqratio -pool_type last&
	# python seqmodel.py -lr $LR -gpu 5 -count 4 -model_type LSTM -seq_ratio $seqratio -pool_type last&

	wait
done

