import numpy as np
import pandas as pd
import clean_CV
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn import datasets
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict

testfile = "../datasets/membrane-beta_4state.3line.txt"  
smalltestfile = "../datasets/parsetest"    
    
def gridscore(infile):
    
    for window_input in range(9,25,2):
        AA_array, Top_array = clean_CV.inputSVM(testfile, window_input)
        X_train, Y_train = AA_array, Top_array
        C_range = [1, 5, 10]
        g_range = [0.001, 0.01]
        param = {'C' : C_range, 'gamma' : g_range}
        clf = GridSearchCV(SVC(), param, n_jobs=-1, cv=3, verbose=True, error_score=np.NaN, return_train_score=False)
        clf.fit(X_train, Y_train)
        df = pd.DataFrame(clf.cv_results)
        filename = str(win_input) + '.csv'
        df.to_csv(filename, sep='\t', encoding='UTF-8')
    
if __name__ == '__main__':
    gridscore(smalltestfile)
    
  
