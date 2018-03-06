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

testfile = "../datasets/membrane-beta_4state.3line.txt"
test = "../datasets/PDBtest.txt"
saved_model = joblib.load('TTmodel.sav')

'''clean_CV.inputSVM(test)
saved_model = joblib.load('TTmodel.sav')
z = np.load('SVM_test.npz')
result = saved_model.predict(z)
print(result)'''

def inputSVM(infile, window_input):
    
    listID = []
    listaa = []
    listTop = []
    final_AAlist = [] 
    final_Toplist = [] 
    listaa_window = []

  
    with open(infile) as pf:
        lines = [line.strip() for line in pf]
    listID = lines[0::3]
    listaa = lines[1::3]
    
    x = window_input//2
    
    for zeroseq in listaa:
        zeroseq = ((x)*'0')+zeroseq+((x)*'0')

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
                 
    return final_AAlist
    
def pred(infile2): 
    Top_output = []   
    z = saved_model.predict(infile2)
    result = z
    
    for element in result:
        Top_output.append(structure_decode_dict[element])
    s = ", "  
    decode = s.join(Top_output)  
    return decode
    
    
if __name__ == '__main__':
    #print(inputSVM(test, 17))
    print(pred(test))
   

