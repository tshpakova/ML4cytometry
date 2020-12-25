
import pandas as pd
import numpy as np
import time 
import matplotlib.pyplot as plt


import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

# import R's utility package
utils = rpackages.importr('utils')

# select a mirror for R packages
utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# R package names
packnames = ('ggplot2', 'hexbin', 'car')

# R vector of strings
from rpy2.robjects.vectors import StrVector

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = packnames
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install)) # add another cran

# Commented out IPython magic to ensure Python compatibility.
# enables the %%R magic, not necessary if you've already done this
# %load_ext rpy2.ipython

# Commented out IPython magic to ensure Python compatibility.
# %%R
# imputation2 <- function(Y,X, item) {
#   
#   
#   model = lm(Y~X[,item])
#   return(model)
# 
# }

fdf = pd.read_excel('full_data.xlsx')
fdf['num'] = np.nan
fdf

fdf['num'] = np.nan

fdf.loc[fdf.Acronyme == 'lean13' ,'num'] = 1
fdf.loc[fdf.Acronyme == 'lean4' ,'num'] = 2
fdf.loc[fdf.Acronyme == 'lean3' ,'num'] = 3
fdf.loc[fdf.Acronyme == 'lean1' ,'num'] = 4
fdf.loc[fdf.Acronyme == 'lean2' ,'num'] = 5
fdf.loc[fdf.Acronyme == 'lean6' ,'num'] = 6
fdf.loc[fdf.Acronyme == 'lean7' ,'num'] = 7
fdf.loc[fdf.Acronyme == 'lean8' ,'num'] = 8
fdf.loc[fdf.Acronyme == 'lean9' ,'num'] = 9
fdf.loc[fdf.Acronyme == 'lean10' ,'num'] = 10
fdf.loc[fdf.Acronyme == 'lean11', 'num'] = 11
fdf.loc[fdf.Acronyme == 'lean12', 'num'] = 12
fdf.loc[fdf.Acronyme == 'lean16_OW', 'num'] = 13
fdf.loc[fdf.Acronyme == 'lean17', 'num'] = 14
fdf.loc[fdf.Acronyme == 'lean18', 'num'] = 15
fdf.loc[fdf.Acronyme == 'lean19', 'num'] = 16
fdf.loc[fdf.Acronyme == 'lean20', 'num'] = 17
fdf.loc[fdf.Acronyme == 'lean21_OW', 'num'] = 18
fdf.loc[fdf.Acronyme == '10MC0704_OW', 'num'] = 19
fdf.loc[fdf.Acronyme == '14MC0963_OW', 'num'] = 20
fdf.loc[fdf.Acronyme == 'lean23_', 'num'] = 21
fdf.loc[fdf.Acronyme == 'lean24', 'num'] = 22
fdf.loc[fdf.Acronyme == 'lean_33', 'num'] = 23
fdf.loc[fdf.Acronyme == 'lean_34', 'num'] = 24
fdf.loc[fdf.Acronyme == 'lean_35', 'num'] = 25
fdf.loc[fdf.Acronyme == 'lean_36_UW', 'num'] = 26
fdf.loc[fdf.Acronyme == 'lean_25', 'num'] = 27
fdf.loc[fdf.Acronyme == 'lean_26', 'num'] = 28
fdf.loc[fdf.Acronyme == 'lean_27', 'num'] = 29
fdf.loc[fdf.Acronyme == 'lean_29', 'num'] = 30
fdf.loc[fdf.Acronyme == 'lean_30_OW', 'num'] = 31
fdf.loc[fdf.Acronyme == 'lean_31', 'num'] = 32
fdf.loc[fdf.Acronyme == 'lean_32', 'num'] = 33
#OB
fdf.loc[fdf.Acronyme == 'ANTAL', 'num'] = 34
fdf.loc[fdf.Acronyme == 'CHAST', 'num'] = 35
fdf.loc[fdf.Acronyme == 'CROTE', 'num'] = 36
fdf.loc[fdf.Acronyme == 'JERAN', 'num'] = 37
fdf.loc[fdf.Acronyme == 'LEDDA', 'num'] = 38
fdf.loc[fdf.Acronyme == 'NEDAN', 'num'] = 39
fdf.loc[fdf.Acronyme == 'NEMNA', 'num'] = 40
fdf.loc[fdf.Acronyme == 'SANNA', 'num'] = 41
fdf.loc[fdf.Acronyme == 'SEPCL', 'num'] = 42
fdf.loc[fdf.Acronyme == 'AMRFA', 'num'] = 43
fdf.loc[fdf.Acronyme == 'BELLY', 'num'] = 44
fdf.loc[fdf.Acronyme == 'BOICH', 'num'] = 45
fdf.loc[fdf.Acronyme == 'CHAFL', 'num'] = 46
fdf.loc[fdf.Acronyme == 'CLASA', 'num'] = 47
fdf.loc[fdf.Acronyme == 'DJOYA', 'num'] = 48
fdf.loc[fdf.Acronyme == 'ELSMA', 'num'] = 49
fdf.loc[fdf.Acronyme == 'ESCAU', 'num'] = 50
fdf.loc[fdf.Acronyme == 'MARMI', 'num'] = 51
fdf.loc[fdf.Acronyme == 'MRIHA', 'num'] = 52
fdf.loc[fdf.Acronyme == 'NZEAF', 'num'] = 53
fdf.loc[fdf.Acronyme == 'REUAU', 'num'] = 54
fdf.loc[fdf.Acronyme == 'SCOSO', 'num'] = 55
fdf.loc[fdf.Acronyme == 'BARDO', 'num'] = 56
fdf.loc[fdf.Acronyme == 'CATNA', 'num'] = 57
fdf.loc[fdf.Acronyme == 'CAVAN', 'num'] = 58
fdf.loc[fdf.Acronyme == 'CORCL', 'num'] = 59
fdf.loc[fdf.Acronyme == 'DACMA', 'num'] = 60
fdf.loc[fdf.Acronyme == 'FERJA', 'num'] = 61
fdf.loc[fdf.Acronyme == 'GILCL', 'num'] = 62
fdf.loc[fdf.Acronyme == 'HALMA', 'num'] = 63
fdf.loc[fdf.Acronyme == 'LANLI', 'num'] = 64
fdf.loc[fdf.Acronyme == 'LECNA', 'num'] = 65
fdf.loc[fdf.Acronyme == 'METJA', 'num'] = 66
fdf.loc[fdf.Acronyme == 'MOUCA', 'num'] = 67
fdf.loc[fdf.Acronyme == 'MOUSE', 'num'] = 68
fdf.loc[fdf.Acronyme == 'NARPA', 'num'] = 69
fdf.loc[fdf.Acronyme == 'NUYMA', 'num'] = 70
fdf.loc[fdf.Acronyme == 'PHISE', 'num'] = 71
fdf.loc[fdf.Acronyme == 'PLAKA', 'num'] = 72
fdf.loc[fdf.Acronyme == 'POISE', 'num'] = 73
fdf.loc[fdf.Acronyme == 'RABRO', 'num'] = 74
fdf.loc[fdf.Acronyme == 'TAILY', 'num'] = 75
fdf.loc[fdf.Acronyme == 'VILDI', 'num'] = 76
# obd
fdf.loc[fdf.Acronyme == 'ADAAH', 'num'] = 77
fdf.loc[fdf.Acronyme == 'DUCGE', 'num'] = 78
fdf.loc[fdf.Acronyme == 'GONPA', 'num'] = 79
fdf.loc[fdf.Acronyme == 'GUIJO', 'num'] = 80
fdf.loc[fdf.Acronyme == 'JEGNA', 'num'] = 81
fdf.loc[fdf.Acronyme == 'MARJO', 'num'] = 82
fdf.loc[fdf.Acronyme == 'MNASA', 'num'] = 83
fdf.loc[fdf.Acronyme == 'RODSH', 'num'] = 84
fdf.loc[fdf.Acronyme == 'AMINA', 'num'] = 85
fdf.loc[fdf.Acronyme == 'AMOCE', 'num'] = 86
fdf.loc[fdf.Acronyme == 'BLACA', 'num'] = 87
fdf.loc[fdf.Acronyme == 'DALJU', 'num'] = 88
fdf.loc[fdf.Acronyme == 'DRICO', 'num'] = 89
fdf.loc[fdf.Acronyme == 'EISMA', 'num'] = 90
fdf.loc[fdf.Acronyme == 'EREJA', 'num'] = 91
fdf.loc[fdf.Acronyme == 'GENCA', 'num'] = 92
fdf.loc[fdf.Acronyme == 'GREPH', 'num'] = 93
fdf.loc[fdf.Acronyme == 'KESJA', 'num'] = 94
fdf.loc[fdf.Acronyme == 'MILMO', 'num'] = 95
fdf.loc[fdf.Acronyme == 'PIJCY', 'num'] = 96
fdf.loc[fdf.Acronyme == 'SALBR', 'num'] = 97
fdf.loc[fdf.Acronyme == 'SQUCA', 'num'] = 98
fdf.loc[fdf.Acronyme == 'ABDPA', 'num'] = 99
fdf.loc[fdf.Acronyme == 'BARJO', 'num'] = 100
fdf.loc[fdf.Acronyme == 'BENTA', 'num'] = 101
fdf.loc[fdf.Acronyme == 'CAMMA', 'num'] = 102
fdf.loc[fdf.Acronyme == 'CHAMI', 'num'] = 103
fdf.loc[fdf.Acronyme == 'CLANI', 'num'] = 104
fdf.loc[fdf.Acronyme == 'DASJE', 'num'] = 105
fdf.loc[fdf.Acronyme == 'DAUSY', 'num'] = 106
fdf.loc[fdf.Acronyme == 'DUBFR', 'num'] = 107
fdf.loc[fdf.Acronyme == 'EPECH', 'num'] = 108
fdf.loc[fdf.Acronyme == 'HAMMY', 'num'] = 109
fdf.loc[fdf.Acronyme == 'KABMA', 'num'] = 110
fdf.loc[fdf.Acronyme == 'LEMPA', 'num'] = 111
fdf.loc[fdf.Acronyme == 'MAREL', 'num'] = 112
fdf.loc[fdf.Acronyme == 'MATCH', 'num'] = 113
fdf.loc[fdf.Acronyme == 'MATMI', 'num'] = 114
fdf.loc[fdf.Acronyme == 'MICMI', 'num'] = 115
fdf.loc[fdf.Acronyme == 'SAECA', 'num'] = 116
fdf.loc[fdf.Acronyme == 'SANDE', 'num'] = 117
fdf.loc[fdf.Acronyme == 'TEHFA', 'num'] = 118
fdf.loc[fdf.Acronyme == 'BENHA', 'num'] = 119
fdf.loc[fdf.Acronyme == 'LUCFL', 'num'] = 120


all = 120
fdf2 = fdf.sort_values(by = ['num']).head(all)

fdf2['lean'] = np.nan
fdf2.loc[np.array(y) == 0, 'lean'] = fdf2.loc[np.array(y) == 0, 'Age_']

fdf2['ob'] = np.nan
fdf2.loc[np.array(y) == 1, 'ob'] = fdf2.loc[np.array(y) == 1, 'Age_']

fdf2['obd'] = np.nan
fdf2.loc[np.array(y) == 2, 'obd'] = fdf2.loc[np.array(y) == 2, 'Age_']

fdf2

import seaborn as sns
sns.boxplot(data = fdf2[['ob', 'obd', 'lean']]).set_title('Age')
plt.savefig('age_true_classes.pdf')

"""# Ising Model"""

y = []
X = []

#lean
for i in range(1,10):
  print(i)
  A = np.load('Carma_lean/A_lean_00'+ str(i) + '_new.npy')
  X.append(np.ravel(A))
  y.append(0)

for i in range(10,34):
  A= np.load('Carma_lean/A_lean_0'+ str(i) + '_new.npy')
  X.append(np.ravel(A))
  y.append(0)

#ob
for i in range(1,10):
  print(i)
  A = np.load('Carma_ob/A_ob_00'+ str(i) + '_new.npy')
  X.append(np.ravel(A))
  y.append(1)

for i in range(10,44):
  A= np.load('Carma_ob/A_ob_0'+ str(i) + '_new.npy')
  X.append(np.ravel(A))
  y.append(1)

#obd
for i in range(1,10):
  print(i)
  A = np.load('Carma_obd/A_obd_00'+ str(i) + '_new.npy')
  X.append(np.ravel(A))
  y.append(2)

for i in range(10,45):
  A= np.load('Carma_obd/A_obd_0'+ str(i) + '_new.npy')
  X.append(np.ravel(A))
  y.append(2)

Y = np.array(y).copy()
X2 = np.array(X).copy()
print(Y.shape, X2.shape)

# Commented out IPython magic to ensure Python compatibility.
arr = []

arr2 = ['ccr7', 'cd27','cd28','cd45ra','cd57','cd279','KLRG1']
for item in range(1,50):
  print(item)
#   %Rpush Y
#   %Rpush X2


#   %Rpush item
#   %R model = imputation2(Y,X2, item)
#   %R model2 = as.vector(Anova(model)['Pr(>F)'][1,1])
  val = %Rget model2
  print(arr2[(item - 1)//7], '\&', arr2[(item - 1)% 7], '.',item, val[0])
  arr.append(val[0])

# Commented out IPython magic to ensure Python compatibility.
# %%R
# library(car)
# Anova(model)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 40)
skf.get_n_splits(X, y)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import statistics

X = np.array(X)
y = np.array(y)
for n in range(1,8):
  arr = []
  for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf = RandomForestClassifier(max_depth=n, random_state=0)
    clf.fit(X_train, y_train)
    arr.append(accuracy_score(y_test, clf.predict(X_test)))
  print(n,statistics.mean(arr),  statistics.stdev(arr)/np.sqrt(10) )

# Compute confusion matrix
def confusion_matrix(act_labels, pred_labels):
    uniqueLabels = [0,1,2]
    clusters = [0,1,2]
    cm = [[0 for i in range(len(clusters))] for i in range(len(uniqueLabels))]
    for i, act_label in enumerate(uniqueLabels):
        for j, pred_label in enumerate(pred_labels):
            if act_labels[j] == act_label:
                cm[i][pred_label] = cm[i][pred_label] + 1
    return cm

cnf_matrix = np.array([ [15,  17],[28,  58]])

import seaborn as sn
df_cm = pd.DataFrame(cnf_matrix, index = [i for i in ["lean", "ob", "obd"]],
                  columns = [i for i in ["lean", "ob", "obd"]])
plt.figure(figsize = (2.5,1.75))
sn.heatmap(df_cm, annot=True)
plt.savefig('corr_conf_mat.pdf')

n = 6
from sklearn.metrics import plot_confusion_matrix

cnf_matrix = np.array([[0,0,0],[0,0,0],[0,0,0]])
for train_index, test_index in skf.split(X, y):
  #print("TRAIN:", train_index, "TEST:", test_index)
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  clf = RandomForestClassifier(max_depth=n, random_state=0)
  clf.fit(X_train, y_train)
  arr.append(accuracy_score(y_test, clf.predict(X_test)))

  labels = y_test
  pred = clf.predict(X_test)
  cnf_matrix += np.array(confusion_matrix(labels, pred))
  #print('\n'.join([''.join(['{:4}'.format(item) for item in row])
  #      for row in cnf_matrix]))
  
  class_names = ['lean', 'ob', 'obd']
  disp = plot_confusion_matrix(clf, X_test, y_test,
                              display_labels=class_names,
                              cmap=plt.cm.Blues)

pred

from sklearn.cluster import KMeans, SpectralClustering
X = np.array(X)
y = np.array(y)

for seed in range(10):
  clustering = SpectralClustering(n_clusters = 3, random_state=seed).fit(X)
  print(seed, sum(clustering.labels_ == y)/len(clustering.labels_))
  labels = y
  pred = clustering.labels_

  # Create a DataFrame with labels and varieties as columns: df
  df = pd.DataFrame({'Labels': labels, 'Clusters': pred})

  # Create crosstab: ct
  ct = pd.crosstab(df['Labels'], df['Clusters'])

  # Display ct
  print(ct)

pred = clustering.labels_

fdf2['class_0'] = np.nan
fdf2.loc[np.array(pred) == 0, 'class_0'] = fdf2.loc[np.array(pred) == 0, 'Age_']

fdf2['class_1'] = np.nan
fdf2.loc[np.array(pred) == 1, 'class_1'] = fdf2.loc[np.array(pred) == 1, 'Age_']

fdf2['class_2'] = np.nan
fdf2.loc[np.array(pred) == 2, 'class_2'] = fdf2.loc[np.array(pred) == 2, 'Age_']

import seaborn as sns
sns.boxplot(data = fdf2[['class_0', 'class_1', 'class_2']]).set_title('Age')
plt.savefig('age_sc.pdf')

labels = y
pred = clustering.labels_
cnf_matrix = confusion_matrix(labels, pred)
print('\n'.join([''.join(['{:4}'.format(item) for item in row])
      for row in cnf_matrix]))

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'Labels': labels, 'Clusters': pred})

# Create crosstab: ct
ct = pd.crosstab(df['Labels'], df['Clusters'])

# Display ct
print(ct)

y = []
X = []

#lean
for i in range(1,10):
  print(i)
  A = np.load('Carma_lean/corr_lean_00'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(0)

for i in range(11,34):
  A= np.load('Carma_lean/corr_lean_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(0)

#ob
for i in range(1,10):
  print(i)
  A = np.load('Carma_ob/corr_ob_00'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(1)

for i in range(10,44):
  A= np.load('Carma_ob/corr_ob_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(1)

#obd
for i in range(1,10):
  print(i)
  A = np.load('Carma_obd/corr_obd_00'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(2)

for i in range(10,23):
  A= np.load('Carma_obd/corr_obd_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(2)

for i in range(24,45):
  A= np.load('Carma_obd/corr_obd_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(2)

from sklearn.cluster import KMeans

for seed in range(10):
  clustering = KMeans(n_clusters = 3, random_state=seed).fit(X)
  print(seed, sum(clustering.labels_ == y)/len(clustering.labels_))
  labels = y
  pred = clustering.labels_

  # Create a DataFrame with labels and varieties as columns: df
  df = pd.DataFrame({'Labels': labels, 'Clusters': pred})

  # Create crosstab: ct
  ct = pd.crosstab(df['Labels'], df['Clusters'])

  # Display ct
  print(ct)

y = []
X = []

#lean
for i in range(1,10):
  print(i)
  A = np.load('Carma_lean/glasso_lean_00'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(0)

for i in range(11,34):
  A= np.load('Carma_lean/glasso_lean_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(0)

#ob
for i in range(1,10):
  print(i)
  A = np.load('Carma_ob/glasso_ob_00'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(1)

for i in range(10,44):
  A= np.load('Carma_ob/glasso_ob_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(1)

#obd
for i in range(1,10):
  print(i)
  A = np.load('Carma_obd/glasso_obd_00'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(2)

for i in range(10,23):
  A= np.load('Carma_obd/glasso_obd_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(2)

for i in range(24,45):
  A= np.load('Carma_obd/glasso_obd_0'+ str(i) + '.npy')
  X.append(np.ravel(A))
  y.append(2)


from sklearn.cluster import KMeans

for seed in range(10):
  clustering = KMeans(n_clusters = 3, random_state=seed).fit(X)
  print(seed, sum(clustering.labels_ == y)/len(clustering.labels_))
  labels = y
  pred = clustering.labels_

  # Create a DataFrame with labels and varieties as columns: df
  df = pd.DataFrame({'Labels': labels, 'Clusters': pred})

  # Create crosstab: ct
  ct = pd.crosstab(df['Labels'], df['Clusters'])

  # Display ct
  print(ct)

y = []
X = []

#lean
for i in range(1,10):
  print(i)
  A = np.load('Carma_lean/miic_lean_00'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)

  X.append(np.ravel(A))
  y.append(0)

for i in range(11,34):
  A= np.load('Carma_lean/miic_lean_0'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)

  X.append(np.ravel(A))
  y.append(0)

#ob
for i in range(1,10):
  print(i)
  A = np.load('Carma_ob/miic_ob_00'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)
  X.append(np.ravel(A))
  y.append(1)

for i in range(10,44):
  A= np.load('Carma_ob/miic_ob_0'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)
  X.append(np.ravel(A))
  y.append(1)

#obd
for i in range(1,10):
  print(i)
  A = np.load('Carma_obd/miic_obd_00'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)
  X.append(np.ravel(A))
  y.append(2)

for i in range(10,23):
  A= np.load('Carma_obd/miic_obd_0'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)
  X.append(np.ravel(A))
  y.append(2)

for i in range(24,45):
  A= np.load('Carma_obd/miic_obd_0'+ str(i) + '.npy')
  while A.shape[0] != 21:
    A = np.append(A,0)
  X.append(np.ravel(A))
  y.append(2)


from sklearn.cluster import KMeans

for seed in range(10):
  clustering = KMeans(n_clusters = 3, random_state=seed).fit(X)
  print(seed, sum(clustering.labels_ == y)/len(clustering.labels_))
  labels = y
  pred = clustering.labels_

  # Create a DataFrame with labels and varieties as columns: df
  df = pd.DataFrame({'Labels': labels, 'Clusters': pred})

  # Create crosstab: ct
  ct = pd.crosstab(df['Labels'], df['Clusters'])

  # Display ct
  print(ct)

X

