import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from numpy import array 
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
   
testfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/parsetest"    

###################################################################################
# Convert into SVM input vector (numerical input) - Steps to follow
###################################################################################


                    
###################################################################################
# Convert into SVM input vector: Sliding Window Input
###################################################################################

listID = []
listaa = []
listTop = []
final_AAlist = [] 
final_Toplist = [] 
listaa_window = []

with open(testfile) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::3]
listaa = lines[1::3]
listTop = lines[2::3]
    
    
#window_input = 0
#while window_input%2==0:
    #window_input = int(input("Window size (must be odd number): "))
window_input=17
x = window_input//2

###################################################################################
# Convert into SVM input vector: Sliding Window Input for AA sequence
###################################################################################

for zeroseq in listaa:
    zeroseq = ((x)*'0')+zeroseq+((x)*'0')

    for aa in range(0,len(zeroseq)):
        window=zeroseq[aa:aa+window_input]
        if len(window)==window_input:
            listaa_window.append(window)

###################################################################################
# Convert into SVM input vector: AA sequence to binary
###################################################################################

for aa in listaa_window:    
    AAlist = [] 
    for ch in aa: 

        if ch in AA_seq_dict.keys():
        
            AAlist.extend(AA_seq_dict[ch]) 
        if ch == '0':
            AAlist.extend(AA_seq_dict['X'])

    final_AAlist.append(AAlist) 
#print (final_AAlist)      
###################################################################################
# Convert into SVM input vector: topology to numerical
###################################################################################
for ch in listTop:
    for x in ch:
        final_Toplist.extend(structure_dict[x]) 
        

    
         

x, y = final_AAlist, final_Toplist
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
clf_model = svm.SVC(gamma=0.01, kernel = 'rbf', C=1.0, class_weight = "balanced").fit(X_train, Y_train)
prediction = clf_model.predict(X_test)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

X, y = final_AAlist, final_Toplist

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.01$)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.01)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
'''
title = "Learning Curves (SVM, linear kernel, $\gamma=0.01$)"
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.01)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)

title = "Learning Curves (SVM, RBF kernel, $\gamma=0.01$)" # PSSM as input
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.01)
plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)'''

plt.show()
