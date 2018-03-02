import numpy as np
from numpy import array 
from numpy import argmax #not in use
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split #not in use
from sklearn import datasets
from sklearn.externals import joblib
import time

testfile = "../datasets/membrane-beta_4state.3line.txt"

start = time.time()

###################################################################################
# Opens the file and appends each feature of the file into 3 lists then dictionary
###################################################################################
listID = []
listaa = []
listTop = []

with open(testfile) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::3]
listaa = lines[1::3]
listTop = lines[2::3]

#print (len(listID))    
    
###################################################################################
# Convert into SVM input vector (numerical input) - Steps to follow
###################################################################################

#AA letters: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y

AA_seq_dict={   'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 
                'C':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'D':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'E':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'F':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'G':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'H':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                'I':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
                'K':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                'L':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                'M':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                'N':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
                'P':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                'Q':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                'R':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                'S':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
                'T':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                'V':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                'W':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
                'Y':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                'X':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]} #extra residue for window

structure_dict = {'i':[0], #0 inner 
                  'P':[1], # 1
                  'L':[2], # 2
                  'o':[3]} #3 outer 
                
###################################################################################
# Convert into SVM input vector: Sliding Window Input
###################################################################################

final_AAlist = [] 
finalized_AAlist = [] 
final_Toplist = [] #final

listaa_window = []
listTop_window = []
    
window_input = 0
while window_input%2==0:
    window_input = int(input("Window size (must be odd number): "))

x = window_input//2

###################################################################################
# Convert into SVM input vector: Sliding Window Input for AA sequence
###################################################################################

for zeroseq in listaa:
    zeroseq = ((x)*'0')+zeroseq+((x)*'0')

    for aa in range(0,len(zeroseq)):
        window=zeroseq[aa:aa+window_input]
        if len(window)==window_input:
            listaa_window.append(window)

###################################################################################
# Convert into SVM input vector: AA sequence to binary
###################################################################################

for aa in listaa_window:    
    AAlist = [] 
    for ch in aa: 
        if ch in AA_seq_dict.keys():
            AAlist.extend(AA_seq_dict[ch]) 
        if ch == '0':
            AAlist.extend(AA_seq_dict['X'])
    final_AAlist.append(AAlist) 
            
###################################################################################
# Convert into SVM input vector: topology to numerical
###################################################################################
for ch in listTop:
    for x in ch:
        final_Toplist.extend(structure_dict[x])      

###################################################################################
# Double check that windows=features (used @dmtr13's code) to check
###################################################################################    
countFeat = 0
for i in final_Toplist:
    countFeat += 1
countAA = 0
for i in final_AAlist:
    countAA += 1
if countFeat != countAA:
    print ("Windows",countAA, "!= Features",countFeat)
else:
    print("You're good!")

###################################################################################
# Predict/Cross-validation (cross_val_score) - Multi-class classification
###################################################################################    

AA_array = np.array(final_AAlist)
Top_array = np.array(final_Toplist)
x,y = AA_array[:-1], Top_array[:-1] #testing set

clf_model = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0).fit(x,y) 
joblib.dump(clf_model, "TTmodel.pk1")

cross_val = int(input("Fold of cross-validation: "))
print ("Prediction: ", clf_model.predict(AA_array)) 
cvs = cross_val_score(clf_model, AA_array, Top_array, cv = cross_val) #scoring="precision"
avg = np.average(cvs) #np.mean
return avg

end = time.time()
time = end-start
print(time)


 
