import os
import os.path
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import tree
import clean_CV
import final_predictor
import all_pssm
from AA_dictionary import AA_seq_dict
from structure_dict import structure_dict
from structure_decode_dict import structure_decode_dict
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets



actualfile = "../datasets/membrane-beta_4state.3line1.txt"
file1 = "../datasets/50unique.3line.txt"

test1 = "../PSI-BLAST/PSSM_50test" #50 sample file to predict topology
PSSM_containingfile = "../PSI-BLAST/PSSM"



#2 lists needed
listID = []
listTop = []

with open(actualfile) as pf:
    lines = [line.strip() for line in pf]
listID = lines[0::3]
listTop = lines[2::3]


#extract matrix needed for each file in my PSSM directory
final_pssmlist = []
for filename in os.listdir(PSSM_containingfile):
    path_open = os.path.join(PSSM_containingfile, filename)
    
    if filename.endswith(".pssm"): 
        
        parsed_pssm = (np.genfromtxt(path_open, skip_header=3, skip_footer=5, autostrip=True, usecols=range(22,42)))/100
        final_pssmlist.append(parsed_pssm) 
        
#PSSM window sized list

main_list = []
window_input = 21
pad = window_input//2
zero = np.zeros(20, dtype=int)

for element in final_pssmlist:
    win_list = []
    for array in range(0, len(element)):
        temp_window =[]
        if array <= 0:
            seq_window = element[(array):(array+pad+1)]
            diff = window_input-len(seq_window)
            for i in range(0, diff):
                temp_window.append(zero)
            temp_window.extend(seq_window)   
        elif array > 0 and array < pad: 
            seq_window = element[0:(array+pad+1)]
            diff = window_input-len(seq_window)
            for i in range(0, diff):
                temp_window.append(zero)
            temp_window.extend(seq_window)   
        elif array >= pad:
            seq_window3 = element[(array-pad):(array+pad+1)]
            if len(seq_window) == window_input:  
                temp_window.extend(seq_window)    
            if len(seq_window) < window_input: 
                diff = window_input-len(seq_window)
                temp_window.extend(seq_window)
                for i in range(0, diff):
                    temp_window.append(zero)
        temp_window = np.array(temp_window)
        final_window = temp_window.flatten()
        win_list.append(final_window)
    main_list.extend(win_list)
templist = np.array(main_list)

#Encoded topology list needed

final_Toplist = []
for ch in listTop:
    for x in ch:
        final_Toplist.extend(structure_dict[x])      
            

x, y = templist, np.array(final_Toplist)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.20, random_state=15)
clf_model = RandomForestClassifier(n_estimators=100, max_features=4, class_weight = "balanced").fit(X_train, Y_train)
prediction = clf_model.predict(X_test)
cm = confusion_matrix(Y_test, prediction)
classes = [0,1,2,3] #i,P,L, o
class_names = 'i','P','L','o'
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

#plot confusion matrix
   
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
plt.savefig("Confusion matrix plots, ws 21")
