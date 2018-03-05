import numpy as np
from sklearn.externals import joblib
from numpy import array
from sklearn import svm

endfile = np.load('testfile.npz')
x = endfile['arr_0.npy']
y = endfile['arr_1.npy']
clf_model = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0).fit(x,y) 

inputfile = 'TTmodel.sav'
joblib.dump(clf_model, 'inputfile')
