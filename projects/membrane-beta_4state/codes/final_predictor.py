from sklearn.externals import joblib
import numpy as np
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
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


testfile = "../datasets/membrane-beta_4state.3line.txt"
test = "../datasets/PDBtest.txt" #one aa seq
test1 = "../datasets/testingdata_pred.txt"
saved_model = joblib.load('TTmodel.sav')

    
listID = []
listaa = []
final_AAlist = [] 
listaa_window = []

all_list = []
filename = open(test1, "r")
filelines = filename.read().splitlines()

with open(test1) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::2]
listaa = lines[1::2]

window_input = 21

x = window_input//2

for zeroseq in listaa:
    zeroseq = ((x)*'0')+zeroseq+((x)*'0')

    for aa in range(0,len(zeroseq)):
        window=zeroseq[aa:aa+window_input]
        
        if len(window)==window_input:
            listaa_window.append(window)
       
#print (listaa_window)

for aa in listaa_window:    
    AAlist = [] 
    for ch in aa: 

        if ch in AA_seq_dict.keys():           
            AAlist.extend(AA_seq_dict[ch]) 
        if ch == '0':
            AAlist.extend(AA_seq_dict['X'])

    final_AAlist.append(AAlist)

Top_output = [] 

z = saved_model.predict(final_AAlist)
result = z

for element in result:
    Top_output.append(structure_decode_dict[element])
    
outputPred = 0
init = 0
#print(all_list)

with open("prediction.txt", "w") as fn:

    for i in range(len(filelines)):
        # id plus sequence imprime les deux 
        if filelines[i].startswith (">"):
            fn.write(filelines[0])
            fn.write("\n")
            fn.write(filelines[1])
            fn.write("\n")
            outputPred = outputPred +len(filelines[i+1])
            x ="".join(Top_output[init:outputPred])
            fn.write(x)
            fn.write("\n")
            init = outputPred
   
#ID_seq_pred = list(zip(listID, listaa, Top_output))    
#print (ID_seq_pred)

with open("prediction_scores_withoutPSSM.txt", "w") as fn:

#Confusion Matrix for SVC
    x, y = final_AAlist, Top_output
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    clf_model = SVC(gamma=0.01, kernel='rbf', C=10, class_weight = "balanced").fit(X_train, Y_train)
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
    print ("done with cm SVC")
#MCC
    x, y = final_AAlist, Top_output
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=15)
    clf_model = SVC(gamma=0.01, kernel='rbf', C=10, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC = matthews_corrcoef(Y_test, prediction)
    fn.write("Matthews correlation coefficient (SVC model): ")
    fn.write("\n")
    fn.write(str(MC))
    fn.write("\n") 
    fn.write("\n")
    print ("done with mcc SVC")
    
#Confusion Matrix for RF
    x, y = final_AAlist, Top_output
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
    print ("done with cm RF")
    
#random forest
       
    x, y = final_AAlist, Top_output
    clf_model = RandomForestClassifier(n_estimators=100, max_features=4, class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC1 = matthews_corrcoef(Y_test, prediction)
    fn.write("Matthews correlation coefficient (RF model): ")
    fn.write("\n")
    fn.write(str(MC1))
    fn.write("\n")
    fn.write("\n")
    print ("done with mcc RF")
    
#Confusion Matrix for DT
    x, y = final_AAlist, Top_output
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
    print ("done with cm DT")
       
#decision tree
    x, y = final_AAlist, Top_output
    clf_model = tree.DecisionTreeClassifier(class_weight = "balanced").fit(X_train, Y_train)
    prediction = clf_model.predict(X_test)
    MC2 = matthews_corrcoef(Y_test, prediction)
    fn.write("Matthews correlation coefficient (DT model): ") 
    fn.write("\n")
    fn.write(str(MC2))
    fn.write("\n")
    print ("done with mcc DT")
            
            
  

