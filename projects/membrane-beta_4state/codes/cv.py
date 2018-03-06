import clean_CV
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
    
#found that window size:17 gave me the best score
    
testfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/parsetest"  
    
def cross_val(infile):
    
    for window_input in range(3,36,2):
        '''array_data = inputSVM(testfile, window_input)
        AA_array = array_data[0] 
        Top_array = array_data[1]'''
        AA_array, Top_array = clean_CV.inputSVM(testfile, window_input)
        
        #cross_val = int(input("Fold of cross-validation: "))
        clf_model = svm.SVC()
        cvs = cross_val_score(clf_model, AA_array, Top_array, cv = 5, verbose=True, n_jobs=-1) 
        avg = np.average(cvs) 
        print ("window size:", window_input, "score:", avg)
    
if __name__ == '__main__':
    cross_val(testfile)
    
    
