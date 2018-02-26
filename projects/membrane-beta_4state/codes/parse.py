import numpy as np
from numpy import array
from numpy import argmax
from sklearn import svm
from sklearn import preprocessing
#rom sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer

###################################################################################
# Opens the file and appends each feature of the file into 3 lists then dictionary
###################################################################################

testfile = "D:/SU-project/projects/membrane-beta_4state/datasets/parsetest"

def parseBeta(infile):

    dictAll = {}
    listID = []
    listaa = []
    listTop = []
  
    with open(infile) as pf:
        #lines = pf.readlines()
        lines = [line.strip() for line in pf]
    listID = lines[0::3]
    listaa = lines[1::3]
    listTop = lines[2::3]
    
    dictA = dict(zip(listID, listaa))
    dictB = dict(zip(listID, listTop))
    
    for key in dictA.keys() & dictB.keys():
        dictAll[key] = (dictA[key], dictB[key])
        
    #return dictAll
    
    dictaa_seq = dictA.values()
    dicttop = dictB.values()
    
    return dictaa_seq
    
###################################################################################
# Convert into SVM input vector (numerical input) - Steps to follow
###################################################################################
def inputSVM(infile):


#AA letters: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y

    AA_seq_dict={'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], #'' to []
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
            'X':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]} #extra residue for window/ not being used for my new verison of code
    #print (AA_seq_dict)
    structure_dict = {'i':0, #inner
                      'P':1,
                      'L':2,
                      'o':3}#outer
                    
###################################################################################
# Convert into SVM input vector: Sliding Window Input
###################################################################################
    #dictAll = {}
    listID = []
    listaa = []
    listTop = []
    AAlist = [] 
    #Toplist = [] 
    final_AAlist = [] #final
    final_Toplist = [] #final
    #listAA = list(AA_seq)
    listaa_window = []
    listTop_window = []
    encwindow_list = []
  
    with open(infile) as pf:
        #lines = pf.readlines()
        lines = [line.strip() for line in pf]
    listID = lines[0::3]
    listaa = lines[1::3]
    listTop = lines[2::3]
        
    #return listaa    #returns difeerent string sets not sublist
    """for element in dictAll.values():
        aa_seq = element[0]
        topology = element[1]"""
        
    window_input = 0
    while window_input%2==0:
        window_input = int(input("Window size (must be odd number): "))
    
    x = window_input//2
    
###################################################################################
# Convert into SVM input vector: Sliding Window Input for AA sequence
###################################################################################
    print ("Breaking down sequences into windows...")
    
    #for element in listaa:
    for zeroseq in listaa:
        zeroseq = ((x)*'0')+zeroseq+((x)*'0')
        for aa in range(0,len(zeroseq)):
            window=zeroseq[aa:aa+window_input]
            if len(window)==window_input:
                listaa_window.append(window)
    #return listaa_window           
    #key = 0
    #seq[i-x:i+x] 
###################################################################################
# Convert into SVM input vector: AA sequence to binary
###################################################################################
    
    for aa in listaa_window:
            for ch in aa: #in range or enumerate
                if ch in AA_seq_dict.keys():
                #for key in aa_seq:
                    #listaa_window[sequence][aa] = aa_seq[key]
                    AAlist.append(AA_seq_dict[ch]) #[(key)]
                # ** expanding dictionaries into k,v pairs
                #AAlist.extend(AA_seq_dict.get(key))
            final_AAlist.append(np.array(AAlist))
    #return final_AAlist
    
###################################################################################
# Convert into SVM input vector: Sliding Window Input for topology
###################################################################################
    print ("Breaking down topology into windows...")
    
    for lineT in listTop:
        for ch in range (0, len(lineT)):
            window=lineT[ch:ch+window_input]
            if len(window)==window_input:
                listTop_window.append(window)
            
    #return listTop_window  
            
###################################################################################
# Convert into SVM input vector: topology to numerical
###################################################################################
            
    for ch in listTop_window:
        if window_input != 1:
            middle = ch[x]
            final_Toplist.append(structure_dict[middle])
        else:
            final_Toplist.append(structure_dict[ch])
            
    #return final_Toplist            
    
###################################################################################
# One-Hot Encoder
###################################################################################    
    
    for sequence in listaa_window:
        for aa in sequence:
            enc = OneHotEncoder(sparse=False)
            enc_window = enc.fit_transform(aa).toarray()
            encwindow_list.append(enc_window)
        encwindow_list = np.array(encwindow_list)
    return encwindow_list
    
    listTop_window = np.array(listTop_window)
    #return listTop_window
    
    #return encwindow_list, listTop_window
    
    """for value in listaa_window:
        letter=[0 for _ in range(len(AA_seq_dict))]
        letter[value]=1
        encwindow_list.append(letter)
    return encwindow_list"""
    
   #each training example has to be 1D 
###################################################################################
# Cross-validation
###################################################################################    



    
if __name__ == '__main__':

    #print(parseBeta("../datasets/parsetest"))
    print(inputSVM("../datasets/parsetest"))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
