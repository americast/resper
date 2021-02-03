# RESPER: Computationally Modelling Resisting Strategies in Persuasive Conversations

This is the official repository for the paper RESPER, to appear in EACL 2021. The necessary codes are contained in the directory `codes/higru`, and the data and the models are contained in `data/higru_bert_data`.

# Evaluation
In `codes/higru` directory, run `python res.py res` to view all the results of the P4G dataset, and `python res.py neg` to view all the results of the CB dataset. One may modify `eval_here.sh` to generate the results. `train_here.sh` may be used to train the models. It is to be noted that the F1 score results here are the average of the five cross validations, whereas the ones mentioned in the paper contains the entire thing taken together. We also measure the standard deviation in this code.

# Citation
If you use our code or refer our work, please cite as 

```
@article{dutt2021resper,
  title={RESPER: Computationally Modelling Resisting Strategies in Persuasive Conversations},
  author={Dutt, Ritam and Sinha, Sayan and Joshi, Rishabh and Chakraborty, Surya Shekhar and Riggs, Meredith and Yan, Xinru and Bao, Haogang and Ros{\'e}, Carolyn Penstein},
  conference={16th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year={2021}
}
```
