from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from sklearn.externals import joblib
from numpy import array
from sklearn import svm
import final_predictor
import clean_CV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics

testfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/parsetest"

    
for window_input in range(15,26,2):
    AA_array, Top_array = clean_CV.inputSVM(testfile, window_input)
        
    np.set_printoptions(threshold = np.inf) 
    loaded = np.load('SVM_test.npz')
    X = loaded['x'] #x test
    Y = loaded['y'] #y test
    
    clf_model = svm.SVC(kernel = 'linear', C=1.0).fit(X,Y) 
    prediction = clf_model.predict(X) 
    
    print ("Classification report for %s" % clf_model)
    print (metrics.classification_report(Y, prediction))

    print ('Accuracy:', accuracy_score(Y, prediction))
    #print ('F1 score:', f1_score(Y, prediction))
    #print ('Recall:', recall_score(Y, prediction))
    #print ('Precision:', precision_score(Y, prediction))
    #print ('\n clasification report:\n', classification_report(Y,prediction))
    print ('\n confusion matrix:\n',confusion_matrix(Y, prediction))
    
