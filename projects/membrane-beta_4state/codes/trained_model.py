import numpy as np
from sklearn.externals import joblib
from numpy import array
from sklearn import svm

def train(infile):
    loaded = np.load(infile)
    X = loaded['x']
    Y = loaded['y']
    #print (len(X), len(Y))
    clf_model = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0).fit(X,Y) 

    inputfile = 'TTmodel.sav'
    joblib.dump(clf_model, inputfile)
    print ("Model trained!")
if __name__ == '__main__':
    train('SVM_test.npz')
