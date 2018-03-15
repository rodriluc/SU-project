import numpy as np
import os
import os.path
from numpy import array 
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from structure_dict import structure_dict
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score
from sklearn import metrics

actualfile = "../datasets/membrane-beta_4state.3line1.txt"
smalltestfile = "../datasets/smalltest"

testPSSM = "../PSI-BLAST/PSSM/>A5VZA8_PSEP1_sequence.fasta.pssm"
PSSM_containingfile = "../PSI-BLAST/PSSM"

def gen(filename):
#open and parse ID and topology from assigned file

    listID = []
    listTop = []

    with open(actualfile) as pf:
        lines = [line.strip() for line in pf]
    listID = lines[0::3]
    listTop = lines[2::3]
    return listID, listTop
    
def extract():

#extract matrix needed for each file in my PSSM directory
    final_pssmlist = []
    for filename in os.listdir(PSSM_containingfile):
        path_open = os.path.join(PSSM_containingfile, filename)
        #print (item)
        if filename.endswith(".pssm"): #dont need to specify "*.fasta.pssm"
            #print (filename)
            parsed_pssm = (np.genfromtxt(path_open, skip_header=3, skip_footer=5, autostrip=True, usecols=range(22,42)))/100
            final_pssmlist.append(parsed_pssm) #extend, append
            #print (final_pssmlist)
    #print(len(final_pssmlist))
    return final_pssmlist

#add padding by adding arrays containing 20 zeroes

def pssm_window(window_input, final_pssmlist):

    '''window_input = 0
    while window_input%2==0:
        window_input = int(input("Window size (must be odd number): "))'''

    main_list = []
    pad = window_input//2
    zero = np.zeros(20, dtype=int)
    
    for element in final_pssmlist:
        win_list = []
        for array in range(0, len(element)):
            temp_window =[]
            if array <= 0:
                seq_window = element[(array):(array+pad+1)]
                diff = window_input-len(seq_window)
                for i in range(0, diff):
                    temp_window.append(zero)
                temp_window.extend(seq_window)   
            elif array > 0 and array < pad: 
                seq_window = element[0:(array+pad+1)]
                diff = window_input-len(seq_window)
                for i in range(0, diff):
                    temp_window.append(zero)
                temp_window.extend(seq_window)   
            elif array >= pad:
                seq_window3 = element[(array-pad):(array+pad+1)]
                if len(seq_window) == window_input:  
                    temp_window.extend(seq_window)    
                if len(seq_window) < window_input: 
                    diff = window_input-len(seq_window)
                    temp_window.extend(seq_window)
                    for i in range(0, diff):
                        temp_window.append(zero)
            temp_window = np.array(temp_window)
            final_window = temp_window.flatten()
            win_list.append(final_window)
        main_list.extend(win_list)
    templist = np.array(main_list)
    return templist
 
def top_window(window_input):
    final_Toplist = []
    for ch in listTop:
        for x in ch:
            final_Toplist.extend(structure_dict[x])
    #print(final_Toplist) 
    return np.array(final_Toplist)      
    
def pssm_model(filename):
    x, y = templist, final_Toplist
    clf_model = SVC(gamma=0.001, kernel='rbf', C=5.0, class_weight = "balanced").fit(x,y)
    inputfile = 'PSSM_model.sav'
    joblib.dump(clf_model, inputfile)
    print ("Model trained!")
    
def pssm_svm(fold):
    #labels = [0,1,2,3]
    x, y = templist, final_Toplist
    clf_model = SVC(gamma=0.001, kernel='rbf', C=5.0, class_weight = "balanced").fit(x,y)
    prediction = clf_model.predict(templist)
    print ("Classification report for %s" % clf_model)
    print (metrics.classification_report(y, prediction))
    prediction = cross_val_predict(clf_model, x, y, cv=fold)
    cm = confusion_matrix(y, prediction)
    print("Confusion Matrix: \n",cm)
    
def pssm_splitMC():
    x, y = templist, final_Toplist
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    print(Y_test)
    clf_model = SVC(gamma=0.01, kernel='rbf', C=10.0, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC = matthews_corrcoef(Y_test, prediction)
    print("Matthews correlation coefficient: ",MC)
    
def pssm_RFC():
    x, y = templist, final_Toplist
    clf_model = RandomForestClassifier(random_state=15)
    scores = cross_val_score(clf_model, x, y, cv=3, scoring = 'f1', class_weight = "balanced")
    score = np.mean(scores)
    print("Random Forest Classifier: ", score)

def pssm_DT():
    x, y = templist, final_Toplist
    clf_model = tree.DecisionTreeClassifier(random_state=15)
    scores = cross_val_score(clf_model, x, y, cv=3, scoring = 'f1', class_weight = "balanced")
    score = np.mean(scores)
    print("Decision Tree Classifier: ", score)

if __name__ == '__main__':
    listID, listTop = gen(actualfile)
    final_pssmlist = extract()
    templist = pssm_window(17, final_pssmlist) #took out final_pssmlist
    final_Toplist = top_window(17)
    pssm_svm(3)
    #pssm_split()
    #pssm_RFC()
    #pssm_DT()        
    #pssm_model(PSSM_containingfile)  
            


#print(temp_pad) 



