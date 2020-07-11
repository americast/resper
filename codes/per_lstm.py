import pandas as pd
import pudb
import pickle
import math
import sys; sys.path.append('common/');
from helper import *
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report

torch.manual_seed(1)
BATCH_SIZE = 32

DATA_PATH = "../data/"
LIMIT = 5

df = pd.read_csv(DATA_PATH+"/Final_annotations.csv")
f = open("../data/datainfo_dict.pt", "rb")
details = pickle.load(f)
f.close()

idx = 0

train_ids =  details["train_ids"][idx]
test_ids  =  details["test_ids"][idx]

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

y_train = [1 if x in sincere_donors_ids else 0 for x in train_ids]
y_test  = [1 if x in sincere_donors_ids else 0 for x in test_ids]

# conv_ids = set(df["Index"])
df = df[df.B4==1]
labels = set(df["Our Label"])
label_dict = dict(zip(labels, range(len(labels))))

x_train = []

for i, id_here in enumerate(train_ids):
    df_here = df.loc[df.B2 == id_here]
    label_here = np.array([label_dict[x] for x in list(df_here["Our Label"])][:LIMIT])
    b = np.zeros((label_here.size, len(labels)))
    b[np.arange(label_here.size),label_here] = 1
    x_train.append(b)

x_test = []
to_prun = []

for i, id_here in enumerate(test_ids):
    df_here = df.loc[df.B2 == id_here]
    if len(df_here) == 0:
        to_prun.append(i)
        print("Prunned at: "+str(i))
        continue
    label_here = np.array([label_dict[x] for x in list(df_here["Our Label"])][:LIMIT])
    b = np.zeros((label_here.size, len(labels)))
    try:
        b[np.arange(label_here.size),label_here] = 1
    except:
        pu.db
    x_test.append(b)

to_prun.reverse()
for each in to_prun:
    del y_test[each]

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        # embeds = self.word_embeddings(sentence)
        try:
            lstm_out, (hidden_here, _) = self.lstm(sentence.view(LIMIT, BATCH_SIZE, -1))
        except:
            pu.db
        lstm_out = F.relu(lstm_out)
        tag_space = self.hidden2tag(hidden_here[-1])
        tag_scores = F.softmax(tag_space, dim=-1)
        return tag_scores

EMBEDDING_DIM = len(labels)
HIDDEN_DIM = 64

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, None, 2)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

NUM_EPOCH = 2000

for epoch in tqdm(range(NUM_EPOCH)):  # again, normally you would NOT do 300 epochs, it is toy data
    # print(str(epoch + 1)+" out of "+str(NUM_EPOCH))
    losses = []
    for i in range(int(len(x_train) / BATCH_SIZE)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()
        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = prepare_sequence(tags, tag_to_ix)
        sentence = torch.Tensor(x_train[i*BATCH_SIZE:(i + 1)*BATCH_SIZE])
        targets = torch.Tensor(y_train[i*BATCH_SIZE:(i + 1)*BATCH_SIZE])
        targets = targets.long()
        # for i in range(len(targets)):
        #     here = targets[i]
        #     targets[i] = [here for x in range(LIMIT)]
        # targets = torch.Tensor(targets)
        # Step 3. Run our forward pass.
        tags = model(sentence)
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tags, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    print("loss: "+str(sum(losses) / len(losses)))



model.eval()
# See what the scores are after training

# Step 1. Remember that Pytorch accumulates gradients.
# We need to clear them out before each instance
model.zero_grad()
# Step 2. Get our inputs ready for the network, that is, turn them into
# Tensors of word indices.
# sentence_in = prepare_sequence(sentence, word_to_ix)
# targets = prepare_sequence(tags, tag_to_ix)
sentence = torch.Tensor(x_test)
targets = torch.Tensor(y_test)
targets = targets.long()
# for i in range(len(targets)):
#     here = targets[i]
#     targets[i] = [here for x in range(LIMIT)]
# targets = torch.Tensor(targets)
# Step 3. Run our forward pass.
BATCH_SIZE = len(sentence)
tags = model(sentence)
# Step 4. Compute the loss, gradients, and update the parameters by
#  calling optimizer.step()
loss = loss_function(tags, targets)
print("loss: "+str(loss))

print(classification_report(targets, tags.argmax(dim = -1)))

# with torch.no_grad():
#     tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    # print(tag_scores)