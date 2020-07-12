#!bin/bash
# Var assignment
LR=2.5e-4
GPU=3
du=300
dc=300
echo ========= lr=$LR ==============
for iter in 1
do
echo --- $Enc - $Dec $iter ---
python EmoMain.py \
-lr $LR \
-gpu $GPU \
-type higru \
-d_h1 $du \
-d_h2 $dc \
-epochs 50 \
-report_loss 720 \
-data_path data/Persuassion/Persuassion0_data.pt \
-vocab_path data/Persuassion/Persuassion0_vocab.pt \
-emodict_path data/Persuassion/Persuassion0_emodict.pt \
-tr_emodict_path data/Persuassion/Persuassion0_tr_emodict.pt \
-dataset Persuassion0
done
#-embedding Persuasion_embedding.pt