import numpy as np
import os
import os.path


actualfile = "../datasets/membrane-beta_4state.3line.txt"
smalltestfile = "../datasets/smalltest"

testPSSM = "../PSI-BLAST/PSSM/>A5VZA8_PSEP1_sequence.fasta.pssm"
PSSM_containingfile = "../PSI-BLAST/PSSM"
#list_title = []
#list_title = os.listdir(PSSM_containingfile)
#print (list_title)




def format_PSSM(inputfile, window_input):
#extract matrix needed for each file in my PSSM directory
    
    for filename in os.listdir(inputfile):
        path_open = os.path.join(inputfile, filename)
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
                x = window_input//2
                zero = list(np.zeros(20))
                
                for item in pssmlist:
                    for i in range(len(item)):
                        temp_pad = []
                        if i in range(x):
                            zeroseq = [temp_pad.append(zero) for nr in range(i,x)]
                        temp_pad.extend(item[0:i+x+1])

                return temp_pad       
 

            
if __name__ == "__main__":
    print(format_PSSM(PSSM_containingfile, 17)) 
    
    
    
