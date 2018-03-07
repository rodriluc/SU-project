# SU-project

Beta barrel (4 state) individual assignment
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
Using "parsetest" as my test file for predictor. 
