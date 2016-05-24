from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

import pandas as pd
import matplotlib.pyplot as plt
import os

# constants
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'data'))
HOSPITAL_DIR = os.path.join(DATA_DIR, 'heart_disease')

with open(os.path.join(HOSPITAL_DIR, 'processed.cleveland.data')) as infile:
    hospital_data = pd.read_csv(infile, header=None, na_values=['?'])

hospital_data = hospital_data.dropna()
X = hospital_data.iloc[:,:13]
Y = hospital_data.iloc[:,13]

acc = []
pre = []
rec = []
f1 = []

for k in range(1,10):
    x = KNeighborsClassifier(k)
    results_1 = cross_val_score(x, X, Y, cv=10)
    results_2 = cross_val_score(x, X, Y, cv=10, scoring='precision_macro')
    results_3 = cross_val_score(x, X, Y, cv=10, scoring='recall_macro')
    results_4 = cross_val_score(x, X, Y, cv=10, scoring='f1_macro')
    acc.append((k, results_1.mean()))
    pre.append((k, results_2.mean()))
    rec.append((k, results_3.mean()))
    f1.append((k, results_4.mean()))

top_acc = sorted(acc, key=lambda p: p[1], reverse=True)
top_pre = sorted(pre, key=lambda p: p[1], reverse=True)
top_rec = sorted(rec, key=lambda p: p[1], reverse=True)
top_f1 = sorted(f1, key=lambda p: p[1], reverse=True)

print "====== K-value testing report ======"
print "Best K for accuracy: %i (%f)" % top_acc[0]
print "Best K for precision: %i (%f)" % top_pre[0]
print "Best K for recall: %i (%f)" % top_rec[0]
print "Best K for f1: %i (%f)" % top_f1[0]

acc_df = pd.DataFrame(acc, columns=['k',"acc"])
pre_df = pd.DataFrame(pre, columns=['k',"pre"])
rec_df = pd.DataFrame(rec, columns=['k',"rec"])
f1_df = pd.DataFrame(f1, columns=['k',"f1"])
results = acc_df.merge(pre_df,on="k").merge(rec_df,on="k").merge(f1_df,on="k")
results.plot(x='k')
plt.show()
