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
 
"""
- I need to check whether the input integer is odd and >= than 3, for window size. 
- With a window size = input, break down the protein sequence with sliding window maybe overlap. 
- Then map protein into assigned numerical values. 
- Do the same for the features: alpha, beta, and coils. """

#AA letters: A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y

AA_seq={'A':'1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'C':'0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'D':'0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'E':'0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'F':'0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'G':'0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'H':'0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0',
        'I':'0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0',
        'K':'0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0',
        'L':'0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0',
        'M':'0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0',
        'N':'0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0',
        'P':'0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0',
        'Q':'0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0',
        'R':'0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0',
        'S':'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0',
        'T':'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0',
        'V':'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0',
        'W':'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0',
        'Y':'0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1'}
 
def inputSVM(infile):
    #AA letters: "A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y"
    
    mappedprotein = []
    map = {'0':0, 'A':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7,'I':8, 'K':9,'L':10,  'M':11, 'N':12, 'P':13, 'Q':14, 'R':15, 'S':16, 'T':17, 'V':18, 'W':19,'Y':20}
    # 0 to be used for window (to make of equal size)
    dictAA = parseBeta(infile)
    #print (dictAA)
    proNumerical = []
    for element in dictAA:
        for character in element:
            proNumerical_ = map[character]
            proNumerical.append(proNumerical_)
            mappedprotein.append(proNumerical_)
            
    #integer_encoded= [char_int]

    return mappedprotein
 
def onehotEncoder(infile):

    """values = array(inputData)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    
    dv = DictVectorizer(sparse=False)
    file1 = inputSVM(infile)
    df = file1.DataFrame(M).convert_objects(convert_numeric=True)
    x = dv.fit_transform(df)
    print (x)"""
 


###################################################################################
# One-Hot Encoder
###################################################################################
 

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = inputSVM(infile).reshape(len(inputSVM(infile)), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    
if __name__ == '__main__':

    #print(parseBeta("../datasets/parsetest"))
    print(onehotEncoder("../datasets/parsetest"))
    #print(inputSVM("../datasets/parsetest"))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
