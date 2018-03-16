import clean_CV
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
    
#found that window size:17 gave me the best score
    
testfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/parsetest"  
    
def cross_val(infile):
    
    for window_input in range(21,26,2):
        AA_array, Top_array = clean_CV.inputSVM(testfile, window_input)
        
        #cross_val = int(input("Fold of cross-validation: "))
        clf_model = svm.SVC()
        cvs = cross_val_score(clf_model, AA_array, Top_array, cv = 5, verbose=True, n_jobs=-1, scoring='f1_macro') 
        avg = np.average(cvs) 
        print ("window size:", window_input, "score:", avg)
    
if __name__ == '__main__':
    cross_val(testfile)
    
    
