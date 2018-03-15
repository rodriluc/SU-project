from sklearn.metrics import precision_recall_fscore_support as score
import numpy as np
from sklearn.externals import joblib
from numpy import array
from sklearn import svm
import PSSM_predictor
import all_pssm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics

actualfile = "../datasets/membrane-beta_4state.3line.txt"
file1 = "../datasets/50unique.3line.txt"

test1 = "../PSI-BLAST/PSSM_50test" #sample file to predict topology
PSSM_containingfile = "../PSI-BLAST/PSSM"
    
for window_input in range(15,28,2):
    listTop = all_pssm.gen(filename)
    final_pssmlist = all_pssm.extract()
    templist= all_pssm.pssm_window(window_input, final_pssmlist)
    final_Toplist = all_pssm.top_window(window_input)    
    x, y = templist, np.array(final_Toplist)
    clf_model = joblib.load('PSSM_model.sav')
    prediction = clf_model.predict(X) 
    
    print ("Classification report for %s" % clf_model)
    print (metrics.classification_report(Y, prediction))
    print ('Accuracy:', accuracy_score(Y, prediction))
    print ('F1 score:', f1_score(Y, prediction))
    print ('Recall:', recall_score(Y, prediction))
    print ('Precision:', precision_score(Y, prediction))
    print ('\n clasification report:\n', classification_report(Y,prediction))
    print ('\n confusion matrix:\n',confusion_matrix(Y, prediction))
    
