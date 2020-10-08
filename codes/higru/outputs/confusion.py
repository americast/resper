import pandas as pd
import pudb
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import pudb
import numpy as np

f = "bert-higru-f_negotiation"
true = []
pred = []

for i in range(5):
	df = pd.read_csv(f+str(i)+".csv")
	true.extend(list(df["True"]))
	pred.extend(list(df["Pred"]))

k = confusion_matrix(true, pred, labels=list([i for i in range(8)])).astype(np.float32)

print(k)
for i, row in enumerate(k):
	sum_here = sum(row)
	row = row / sum_here
	#pu.db
	k[i] = row

for i in range(8):
	for j in range(8):
		print(k[j][i])
	print()

df_cm = pd.DataFrame(k, index = [i for i in ["not-a-strategy","selective-avoidance","counter-argumentation","personal-choice","source-degradation","information-inquiry","self-pity","self-assertion"]],
                      columns = [i for i in ["not-a-strategy","selective-avoidance","counter-argumentation","personal-choice","source-degradation","information-inquiry","self-pity","self-assertion"]])

plt.figure(figsize = (8,8))
sn.heatmap(df_cm, annot=True)

plt.show()
