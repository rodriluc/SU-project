from sklearn.externals import joblib
import numpy as np
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
from sklearn import svm


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

window_input = 17

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
  

