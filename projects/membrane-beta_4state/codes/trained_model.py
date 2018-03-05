import numpy as np
from sklearn.externals import joblib
from numpy import array
from sklearn import svm

def train(infile):
    endfile = np.load(infile)
    x = endfile['arr_0.npy']
    y = endfile['arr_1.npy']
    #print (len(x), len(y))
    clf_model = svm.SVC().fit(x,y) 
    

    inputfile = 'TTmodel.sav'
    joblib.dump(clf_model, inputfile)
    
if __name__ == '__main__':
    train('testfile.npz')
