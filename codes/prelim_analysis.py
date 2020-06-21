#!/usr/bin/env python
# coding: utf-8

# In[45]:


import sys; sys.path.append('common/');
from helper import *

DATA_PATH = '../data/'

df = pd.read_csv(DATA_PATH+'/Final_annotations.csv')
df = df.rename(columns={"Our Label": "fine_labels"})

for index, row in df.iterrows():
    if row['B4']==1 and row['fine_labels']!= row['fine_labels']:
        print(index, row)


fine_labels = [i for i in list(set(df['fine_labels'])) if i==i]

fine2coarse_dict = {}
fine2coarse_dict['deflect-responsibility'] = 'contesting'
fine2coarse_dict['organization-inquiry']   = 'contesting'#'biased-processing'
fine2coarse_dict['attack-credibility']     = 'contesting'
fine2coarse_dict['self-pity']              = 'empowerment'
fine2coarse_dict['nitpicking']             = 'contesting'
fine2coarse_dict['direct-rejection']       = 'avoidance'
fine2coarse_dict['personal-choice']        = 'empowerment'
fine2coarse_dict['delay-tactic']           = 'avoidance'
fine2coarse_dict['hesitance']              = 'avoidance'
fine2coarse_dict['not-a-strategy']         = 'not-a-strategy'

fine2resistance_dict = {}
fine2resistance_dict['deflect-responsibility'] = 'counter-argumentation'
fine2resistance_dict['organization-inquiry']   = 'source-degradation' # biased-processing
fine2resistance_dict['attack-credibility']     = 'source-degradation'
fine2resistance_dict['self-pity']              = 'empowerment'
fine2resistance_dict['nitpicking']             = 'counter-argumentation'
fine2resistance_dict['direct-rejection']       = 'self-assertion'
fine2resistance_dict['personal-choice']        = 'attitude-bolstering'
fine2resistance_dict['delay-tactic']           = 'avoidance'
fine2resistance_dict['hesitance']              = 'avoidance'
fine2resistance_dict['not-a-strategy']         = 'not-a-strategy'

coarse_labels     = []
resistance_labels = []

for i, row in df.iterrows():
    label = row['fine_labels']
    if label in fine2coarse_dict:
        coarse_labels.append(fine2coarse_dict[label])
        resistance_labels.append(fine2resistance_dict[label])
    else:
        resistance_labels.append('None')
        coarse_labels.append('None')
        
df['coarse_labels'] = coarse_labels
df['resistance_labels'] = resistance_labels

info_df = pd.read_csv(DATA_PATH+'/300_info.csv',sep=',')

sincere_donors_ids=set()
sincere_nondonors_ids=set()
for index, row in info_df.iterrows():
    did= row['B2']
    role=str(row['B4'])
    prop_amt=float(row['B5'])
    amt= float(row['B6'])
    if role =='1':
        if amt>0 and prop_amt<= amt:
            sincere_donors_ids.add(did)
        elif amt==0 and (prop_amt== 0 or math.isnan(prop_amt)):
            sincere_nondonors_ids.add(did)
        elif amt>0 and prop_amt> amt:
            sincere_donors_ids.add(did)
        else:
            # print(amt,prop_amt)
#             sincere_donors_ids.add(did)
            sincere_nondonors_ids.add(did)

print("Sincere donors= ",len(sincere_donors_ids))
print("Sincere non-donors= ",len(sincere_nondonors_ids))


# In[46]:


print('No of valid data-points {}'.format(len(df[df.B4==1])))

import scipy.stats
def print_label_dist(label_name):
    labels_list  = list(set(list(df[df.B4==1][label_name])))
    labels_dict  = ddict(lambda: ddict(int))
    convs_dict   = ddict(lambda: ddict(list))
        
    df2 = df[df.B4 ==1]
    for index, row in df2.iterrows():
        group_name = 'SD'
        if row['B2'] in sincere_nondonors_ids:
            group_name ='SND'
        labels_dict[group_name][row[label_name]]+=1
        
        for label in labels_list:
            if label == row[label_name]:
                convs_dict[group_name][label].append(1)
            else:
                convs_dict[group_name][label].append(0)
    
    print('Distribution of the different strategies among the different categories')
    print("Label\t\t\t#Donor #NonDonor Mean-D Mean-ND pval-1\t pval-2")
    for label in labels_list:
        a = convs_dict['SD'][label]
        b = convs_dict['SND'][label]
        mu_a = round(np.mean(a),3)
        mu_b = round(np.mean(b),3)
        sum_a = sum(a)
        sum_b = sum(b)
        stat, pval   = scipy.stats.ks_2samp(a,b)
        stat2,pval2  = scipy.stats.ttest_ind(a,b,equal_var=False)
        if len(label)<=len('personal-choice'):
            print("{}\t\t{}\t{}\t{}\t{}\t{}\t{}".format(label, sum_a, sum_b, mu_a , mu_b, round(pval,4), round(pval2,4)))
        else:
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}".format(label, sum_a, sum_b, mu_a , mu_b, round(pval,4), round(pval2,4)))
    print('*********************************\n\n')
    return 


# In[ ]:


print_label_dist('fine_labels')

print_label_dist('coarse_labels')

print_label_dist('resistance_labels')


# In[ ]:




