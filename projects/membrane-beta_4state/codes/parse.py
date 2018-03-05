import numpy as np
from numpy import array 
from numpy import argmax #not in use
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets


###################################################################################
# Opens the file and appends each feature of the file into 3 lists then dictionary
###################################################################################

#testfile = "D:/SU-project/projects/membrane-beta_4state/datasets/parsetest"

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
    structure_dict = {'i':[0], #0 inner 
                      'P':[1], # 1
                      'L':[2], # 2
                      'o':[3]} #3 outer 
                    
###################################################################################
# Convert into SVM input vector: Sliding Window Input
###################################################################################
    #dictAll = {}
    listID = []
    listaa = []
    listTop = []
    #AAlist = [] 
    #Toplist = [] 
    final_AAlist = [] 
    finalized_AAlist = [] 
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
    
    #print (len(listID))    
        
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
    #print ("Breaking down sequences into windows...")
    #print(listaa)
    #for element in listaa:

    for zeroseq in listaa:
        zeroseq = ((x)*'0')+zeroseq+((x)*'0')
        #zeroseq = ((x)*(AA_seq_dict['X']))+zeroseq+((x)*(AA_seq_dict['X']))

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
        AAlist = [] 
        for ch in aa: #in range or enumerate
            
            #for loop to specify range for master list
            if ch in AA_seq_dict.keys():
            
            #for key in aa_seq:
                #listaa_window[sequence][aa] = aa_seq[key]
                AAlist.extend(AA_seq_dict[ch]) #[(key)]
            if ch == '0':
                AAlist.extend(AA_seq_dict['X'])
            # ** expanding dictionaries into k,v pairs
            #AAlist.extend(AA_seq_dict.get(key))
        final_AAlist.append(AAlist) #np.asarray
    #print(len(final_AAlist))
    #return final_AAlist
                
###################################################################################
# Convert into SVM input vector: topology to numerical
###################################################################################
    for ch in listTop:
        for x in ch:
            final_Toplist.extend(structure_dict[x]) #append each character as indv string
    #return listTop        
    #return final_Toplist            
    
###################################################################################
# Double check that windows=features (used dmtr13's code) to check
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
    
    
   #each training example has to be 1D
###################################################################################
# Predict/Cross-validation (cross_val_score) - Multi-class classification
###################################################################################    
    #final_AAlist and final_Toplist
    
    AA_array = np.array(final_AAlist)
    Top_array = np.array(final_Toplist)
    x,y = AA_array[:-1], Top_array[:-1] #testing set
    #print (x,y)
    cross_val = int(input("Fold of cross-validation: "))
    clf = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0).fit(x,y) #gamma matters
    print ("Prediction: ", clf.predict(AA_array)) # ([AA_array[-1]])
    
    cvs = cross_val_score(clf, AA_array, Top_array, cv = cross_val) #scoring="precision"
    avg = np.average(cvs) #np.mean
    return avg
    
if __name__ == '__main__':

    #print(parseBeta("../datasets/parsetest"))
    #print(inputSVM("../datasets/parsetest"))
    print(inputSVM("../datasets/membrane-beta_4state.3line.txt"))
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
