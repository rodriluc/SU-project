import numpy as np
from numpy import array 
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
   
testfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/parsetest"    

###################################################################################
# Convert into SVM input vector (numerical input) - Steps to follow
###################################################################################
def inputSVM(infile):
                    
###################################################################################
# Convert into SVM input vector: Sliding Window Input
###################################################################################
    
    listID = []
    listaa = []
    listTop = []
    final_AAlist = [] 
    final_Toplist = [] #final
    listaa_window = []

  
    with open(infile) as pf:
        lines = [line.strip() for line in pf]
    listID = lines[0::3]
    listaa = lines[1::3]
    listTop = lines[2::3]
        
        
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
# Convert into array and save
###################################################################################            
            
    AA_array = np.array(final_AAlist)
    Top_array = np.array(final_Toplist)
    x,y = AA_array, Top_array #testing set
    outfile = 'testfile'
    np.savez(outfile, x, y)
            
    return AA_array, Top_array
    #return len(x), len(y)
###################################################################################
# SVM model prediction
###################################################################################    
    
def SVM(infile): 
    Top_output = []   
    #array_data = inputSVM(infile)
    #x = array_data[0] 
    #y = array_data[1] 
    
    np.set_printoptions(threshold = np.inf) #svm
    endfile = np.load('testfile.npz')
    x = endfile['arr_0.npy']
    y = endfile['arr_1.npy']

    clf_model = svm.SVC(gamma=0.001, kernel = 'linear', C=1.0).fit(x,y) 
    result = clf_model.predict(x) 
    
    for element in result:
        Top_output.append(structure_decode_dict[element])
    s = ", "  
    out = s.join(Top_output)  
    return out
    
###################################################################################
# Predict/Cross-validation (cross_val_score) - Multi-class classification
################################################################################### 
    
def cross_val(infile):
    clf_model = SVM(infile)
    array_data = inputSVM(infile)
    AA_array = array_data[0] 
    Top_array = array_data[1] 
    
    cross_val = int(input("Fold of cross-validation: "))
    cvs = cross_val_score(clf_model, AA_array, Top_array, cv = cross_val) 
    avg = np.average(cvs) 
    return avg
    
if __name__ == '__main__':
    inputSVM(testfile)
    SVM(testfile)
    #print(inputSVM(testfile))
    #print(SVM(smalltestfile))
    #print(cross_val(smalltestfile))
    
