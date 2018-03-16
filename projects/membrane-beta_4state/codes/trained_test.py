import numpy as np
from sklearn.externals import joblib
from numpy import array
from sklearn import datasets
import time
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
from sklearn import svm

testfile = "../datasets/membrane-beta_4state.3line.txt"

start = time.time()

def trained_test(infile):

    listID = []
    listaa = []
    listTop = []

    with open(testfile) as pf:
        lines = [line.strip() for line in pf]
    listID = lines[0::3]
    listaa = lines[1::3]
    listTop = lines[2::3]             

    final_AAlist = [] 
    finalized_AAlist = [] 
    final_Toplist = [] #final

    listaa_window = []
    listTop_window = []

    Top_output = []
        
    window_input = 0
    while window_input%2==0:
        window_input = int(input("Window size (must be odd number): "))

    x = window_input//2 #return topology for each protein 

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
                
    for ch in listTop:
        for x in ch:
            final_Toplist.extend(structure_dict[x])      
        
    AA_array = np.array(final_AAlist)
    Top_array = np.array(final_Toplist)
    x,y = AA_array[:-1], Top_array[:-1] 
    np.savez('testfile.npz', x, y)

    np.set_printoptions(threshold = np.inf) #svm
    endfile = np.load('testfile.npz')
    x = endfile['arr_0.npy']
    y = endfile['arr_1.npy']

    clf_model = svm.SVC(gamma=0.01, kernel = 'linear', C=10).fit(x,y) #train

    result = clf_model.predict(AA_array) #test on training data
    for element in result:
        Top_output.append(structure_decode_dict[element])
    #print ("Prediction: ", Top_output) 

    return (final_AAlist, final_Toplist)
    
    end = time.time()
    time = end-start
    #print(time)
    
if __name__ == '__main__':
    print(trained_test(testfile))
