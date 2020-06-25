# !/bin/bash

# python3 train.py ER/0_ ER/train0.char_to_idx ER/train0.word_to_idx ER/train0.tag_to_idx ER/train0.csv ER/test0.csv 100 1

python3 train.py EE/0_ EE/train0.char_to_idx EE/train0.word_to_idx EE/train0.tag_to_idx EE/train0.csv EE/test0.csv 100 0&
python3 train.py EE/1_ EE/train1.char_to_idx EE/train1.word_to_idx EE/train1.tag_to_idx EE/train1.csv EE/test1.csv 100 1&
python3 train.py EE/2_ EE/train2.char_to_idx EE/train2.word_to_idx EE/train2.tag_to_idx EE/train2.csv EE/test2.csv 100 2&
python3 train.py EE/3_ EE/train3.char_to_idx EE/train3.word_to_idx EE/train3.tag_to_idx EE/train3.csv EE/test3.csv 100 3&
python3 train.py EE/4_ EE/train4.char_to_idx EE/train4.word_to_idx EE/train4.tag_to_idx EE/train4.csv EE/test4.csv 100 4&

python3 train.py ER/0_ ER/train0.char_to_idx ER/train0.word_to_idx ER/train0.tag_to_idx ER/train0.csv ER/test0.csv 100 5&
python3 train.py ER/1_ ER/train1.char_to_idx ER/train1.word_to_idx ER/train1.tag_to_idx ER/train1.csv ER/test1.csv 100 0&
python3 train.py ER/2_ ER/train2.char_to_idx ER/train2.word_to_idx ER/train2.tag_to_idx ER/train2.csv ER/test2.csv 100 1&
python3 train.py ER/3_ ER/train3.char_to_idx ER/train3.word_to_idx ER/train3.tag_to_idx ER/train3.csv ER/test3.csv 100 2&
python3 train.py ER/4_ ER/train4.char_to_idx ER/train4.word_to_idx ER/train4.tag_to_idx ER/train4.csv ER/test4.csv 100 3&

python3 train.py all/0_ all/train0.char_to_idx all/train0.word_to_idx all/train0.tag_to_idx all/train0.csv all/test0.csv 100 0&
python3 train.py all/1_ all/train1.char_to_idx all/train1.word_to_idx all/train1.tag_to_idx all/train1.csv all/test1.csv 100 4&
python3 train.py all/2_ all/train2.char_to_idx all/train2.word_to_idx all/train2.tag_to_idx all/train2.csv all/test2.csv 100 5&
python3 train.py all/3_ all/train3.char_to_idx all/train3.word_to_idx all/train3.tag_to_idx all/train3.csv all/test3.csv 100 3&
python3 train.py all/4_ all/train4.char_to_idx all/train4.word_to_idx all/train4.tag_to_idx all/train4.csv all/test4.csv 100 4&
