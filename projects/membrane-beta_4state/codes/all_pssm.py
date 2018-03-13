import numpy as np
import os
import os.path
from numpy import array 
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
from sklearn.externals import joblib


actualfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/smalltest"

testPSSM = "../PSI-BLAST/PSSM/>A5VZA8_PSEP1_sequence.fasta.pssm"
PSSM_containingfile = "../PSI-BLAST/PSSM"

#open and parse ID and topology from assigned file

listID = []
listTop = []
final_Toplist = [] 

with open(actualfile) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::3]
listTop = lines[2::3]

window_input = 0
while window_input%2==0:
    window_input = int(input("Window size (must be odd number): "))
main_wlist = []

#extract matrix needed for each file in my PSSM directory

for filename in os.listdir(PSSM_containingfile):
    path_open = os.path.join(PSSM_containingfile, filename)
    pssmlist = []
    #print (item)
    if filename.endswith(".pssm"): #dont need to specify "*.fasta.pssm"
        #print (filename)
        with open(path_open) as open_sesame:
            x = open_sesame.read().splitlines()
            #print (x)
        
            for line in x:
                parsed_pssm = (np.genfromtxt(path_open, skip_header=3, skip_footer=5, autostrip=True, usecols=range(22,42)))
                pssmlist.append(parsed_pssm/100)
                #print(pssmlist)
#return pssmlist

#add padding by adding arrays containing 20 zeroes

            zero = np.zeros(20))
            
            for item in pssmlist:
                for i in range(len(item)):
                    templist = []
                    if i in range(window_input):
                        zeroseq = [templist.append(zero) for nr in range(i,window_input)]
                    templist.extend(item[0:i+window_input+1])

            #return templist       
#word list with PSSM
            wlist = []
            for  i in range (0, len(templist)-(window_input-1)):
                x = templist[i:i+(window_input)]
                x = [j for i in x for j in i]
                wlist.append(x)
            main_wlist.append(wlist)
            print(main_wlist)
            


#print(temp_pad) 



