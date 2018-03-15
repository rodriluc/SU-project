import os
import os.path
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import tree
import clean_CV
import final_predictor
import all_pssm
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict


actualfile = "../datasets/membrane-beta_4state.3line.txt"
file1 = "../datasets/50unique.3line.txt"

test1 = "../PSI-BLAST/PSSM_50test" #50 sample file to predict topology
PSSM_containingfile = "../PSI-BLAST/PSSM"

#2 lists needed
listID = []
listTop = []

with open(file1) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::3]
listTop = lines[2::3]


#extract matrix needed for each file in my PSSM directory
final_pssmlist = []
for filename in os.listdir(test1):
    path_open = os.path.join(test1, filename)
    
    if filename.endswith(".pssm"): 
        
        parsed_pssm = (np.genfromtxt(path_open, skip_header=3, skip_footer=5, autostrip=True, usecols=range(22,42)))/100
        final_pssmlist.append(parsed_pssm) 
        
#PSSM window sized list

main_list = []
window_input = 17
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

#Encoded topology list needed

final_Toplist = []
for ch in listTop:
    for x in ch:
        final_Toplist.extend(structure_dict[x])      
        
#PSSM model
x, y = templist, np.array(final_Toplist)
clf_model = joblib.load('PSSM_model.sav')
prediction = clf_model.predict(x)

#predicted output in correct format 
listID_out = []
listaa = []
final_AAlist = [] 
listaa_window = []

all_list = []
filename = open(file1, "r")
filelines = filename.read().splitlines()

with open(file1) as pf:
    lines = [line.strip() for line in pf]
listID_out = lines[0::3]
listaa = lines[1::3]

for zeroseq in listaa:
    zeroseq = ((pad)*'0')+zeroseq+((pad)*'0')

    for aa in range(0,len(zeroseq)):
        window=zeroseq[aa:aa+window_input]
        
        if len(window)==window_input:
            listaa_window.append(window)

for aa in listaa_window:    
    AAlist = [] 
    for ch in aa: 

        if ch in AA_seq_dict.keys():           
            AAlist.extend(AA_seq_dict[ch]) 
        if ch == '0':
            AAlist.extend(AA_seq_dict['X'])

    final_AAlist.append(AAlist)

Top_output = [] 

for element in prediction:
    Top_output.append(structure_decode_dict[element])
    
outputPred = 0
init = 0

with open("PSSM_prediction.txt", "w") as fn:

    for i in range(len(filelines)):
        
        if filelines[i].startswith (">"):
            fn.write(filelines[0])
            fn.write("\n")
            fn.write(filelines[1])
            fn.write("\n predicted topology:\n")
            outputPred = outputPred +len(filelines[i+1])
            x ="".join(Top_output[init:outputPred])
            fn.write(x)
            fn.write("\n")
            init = outputPred
        
print("Done!") 

with open("PSSM_prediction_scores.txt", "w") as fn:

#Confusion Matrix for SVC
    x, y = templist, np.array(final_Toplist)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    clf_model = SVC(gamma=0.01, kernel='rbf', C=10.0, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    fn.write("\n")
    fn.write("Classification report for %s" % clf_model)
    fn.write("\n")
    fn.write(metrics.classification_report(Y_test, prediction))
    fn.write("\n")
    cm = confusion_matrix(Y_test, prediction)
    fn.write("Confusion Matrix SVC: ")
    fn.write("\n")
    fn.write(str(cm))
    fn.write("\n")
    
#MCC
    x, y = templist, np.array(final_Toplist)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    clf_model = SVC(gamma=0.01, kernel='rbf', C=10.0, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC = matthews_corrcoef(Y_test, prediction)
    fn.write("Matthews correlation coefficient (SVC model): ")
    fn.write("\n")
    fn.write(str(MC))
    fn.write("\n") 
    fn.write("\n")
    #use MCC score  for SVC, RF and DT model
    
#Confusion Matrix for RF
    x, y = templist, np.array(final_Toplist)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    clf_model = RandomForestClassifier(n_estimators=100, max_features=4, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    fn.write("Classification report for %s" % clf_model)
    fn.write("\n")
    fn.write(metrics.classification_report(Y_test, prediction))
    fn.write("\n")
    cm = confusion_matrix(Y_test, prediction)
    fn.write("Confusion Matrix RF: ")
    fn.write("\n")
    fn.write(str(cm))
    fn.write("\n")

#random forest
    #print (len(templist), len(np.array(final_Toplist)))    
    x, y = templist, np.array(final_Toplist)
    clf_model = RandomForestClassifier(n_estimators=100, max_features=4, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC1 = matthews_corrcoef(Y_test, prediction)
    fn.write("Matthews correlation coefficient (RF model): ")
    fn.write("\n")
    fn.write(str(MC1))
    fn.write("\n")
    fn.write("\n")
    
#Confusion Matrix for DT
    x, y = templist, np.array(final_Toplist)
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    clf_model = tree.DecisionTreeClassifier(class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    fn.write("Classification report for %s" % clf_model)
    fn.write("\n")
    fn.write(metrics.classification_report(Y_test, prediction))
    fn.write("\n")
    cm = confusion_matrix(Y_test, prediction)
    fn.write("Confusion Matrix DT: ")
    fn.write("\n")
    fn.write(str(cm))
    fn.write("\n")
       
#decision tree
    x, y = templist, np.array(final_Toplist)
    clf_model = tree.DecisionTreeClassifier(class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC2 = matthews_corrcoef(Y_test, prediction)
    fn.write("Matthews correlation coefficient (DT model): ") 
    fn.write("\n")
    fn.write(str(MC2))
    fn.write("\n")
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

