import clean_CV
from sklearn.externals import joblib
import numpy as np
from numpy import array
from sklearn import datasets
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
from sklearn import svm
import trained_model

test = "../datasets/PDBtest.txt"

clean_CV.inputSVM(test)
saved_model = joblib.load('TTmodel.sav')
z = np.load('testfile.npz')
result = saved_model.predict(z)
print(result)


