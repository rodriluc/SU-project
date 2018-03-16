# SU-project

**Beta barrel (4 state) individual assignment**

1. clean_CV includes functions for:
    -parser  
    -window list  
    -binary list of aa  
    -numerical list of topology  
    -topology back to ch  
    -save into .npz file  
    -SVM  
    -cross validation  
    
2. cv
    -tested on a range of window sizes  
    -window size 17  

3. trained_model
    -saves trained model
    
4. final_predictor
    -predicts topology using model  
    -output looks like this for each protein:  
      ID  
      amino acid sequence  
      predicted topology  
      
Trained my model with all the proteins I was given (42 proteins) and window size 17 with a linear kernel.
Used "testingdata_pred.txt" as my test file for predictor. 

**For PSSM portion:**

1. all_pssm includes functions to:  
    -parse and prepare input for predictor  
    -optimized version (PSSM_model.sav)  
    
2. PSSM_predictor  
    -predicts topology on 50 proteins using model  
    -"PSSM_prediction.txt" is the output for the predicted topology of 50 proteins  
    -"PSSM_prediction_scores.txt" is all the data of my scores  
    
Trained my model with the PSSMs found in folder "PSSM" and tested on the PSSMs found in folder "PSSM_50test", both located in "PSI-BLAST".
