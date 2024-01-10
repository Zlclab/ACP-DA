import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix 

def calculate_metric(gt, pred): 
    pred[pred>0.5]=1
    pred[pred<1]=0
    confusion = confusion_matrix(gt,pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    return TN / float(TN+FP)

AA = 'ACDEFGHIKLMNPQRSTVWY'
maxlen = 50

def Prepare_data(path):
    seq = []
    label = []
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[2]
                if label_temp == '1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq.append(line[:-1])
    return seq, label

def Kmers_funct(seq, size): 
    return [seq[x:x+size] for x in range(len(seq) - size + 1)]

##############################################################Ordinal encoding (OE)

def OE(seq_list):
    oe = {'A': 1,'C': 2,'D': 3,'E': 4,'F': 5,'G': 6,'H': 7,'I': 8,'K': 9,'L': 10,'M': 11,
          'N': 12,'P': 13,'Q': 14,'R': 15,'S': 16,'T': 17,'V': 18,'W': 19,'Y': 20}
    
    result = []
    for i in range(len(seq_list)):
        m = []
        for j in range(len(seq_list[i])):
            m.append(oe.get(seq_list[i][j]))
        result.append(m)
    return [result[i]+ [0] * (maxlen - len(result[i])) for i in range(len(result))]

########################################################Amino acid composition (AAC)

def AAC(seq_list):
    
    result = []
    for i in range(len(seq_list)):
        m = [0] * len(AA)
        for j in range(len(AA)):
            m[j] = seq_list[i].count(AA[j])/len(seq_list[i])
        result.append(m)
    return result

#######################################################Dipeptide composition (DPC)

def reverse_string(input_str):
    return input_str[::-1]

def DPC(seq_list):
    comb = []
    for i in range(len(AA)):
        for j in range(i,len(AA)):
            comb.append(AA[i]+AA[j])
    comb_ = [reverse_string(i) for i in comb]
    
    result = []
    for i in range(len(seq_list)):
        k = Kmers_funct(seq_list[i],2)
        m = [0] * len(comb)
        for j in range(len(comb)):
            m[j] = (k.count(comb[j]) + k.count(comb_[j]))/len(k)
        result.append(m)
    return result
    

###########################################Grouped Amino Acids composition (GAAC) 

def GAAC(seq_list):
    group_list = ['θ', 'λ', 'μ', 'ξ', 'φ']
    group = {'A': 'θ','C': 'φ','D': 'ξ','E': 'ξ','F': 'λ','G': 'θ','H': 'μ','I': 'θ','K': 'μ','L': 'θ',
             'M': 'θ','N': 'φ','P': 'φ','Q': 'φ','R': 'μ','S': 'φ','T': 'φ','V': 'θ','W': 'λ','Y': 'λ'}
    result = []
    for i in range(len(seq_list)):
        group_seq = [group.get(seq_list[i][n]) for n in range(len(seq_list[i]))]
        m = [0] * len(group_list)
        for j in range(len(group_list)):
            m[j] = group_seq.count(group_list[j])/len(seq_list[i])
        result.append(m)
    return result
#############################################Grouped dipeptide composition (GDPC) 

def GDPC(seq_list):
    group_list = ['θ', 'λ', 'μ', 'ξ', 'φ']
    group_comb = [g1 + g2 for g1 in group_list for g2 in group_list]            
    group = {'A': 'θ','C': 'φ','D': 'ξ','E': 'ξ','F': 'λ','G': 'θ','H': 'μ','I': 'θ','K': 'μ','L': 'θ',
             'M': 'θ','N': 'φ','P': 'φ','Q': 'φ','R': 'μ','S': 'φ','T': 'φ','V': 'θ','W': 'λ','Y': 'λ'}
    result = []
    for i in range(len(seq_list)):
        group_seq = [group.get(seq_list[i][n]) for n in range(len(seq_list[i]))]
        two_mers = Kmers_funct("".join(group_seq),2)
        m = [0] * len(group_comb)
        for j in range(len(group_list)):
            m[j] = two_mers.count(group_comb[j])/(len(seq_list[i])-1)
        result.append(m)
    return result

############################################Grouped Tripeptide Composition (GTPC) 

def GTPC(seq_list):
    group_list = ['θ', 'λ', 'μ', 'ξ', 'φ']
    group_comb = [g1 + g2 + g3 for g1 in group_list for g2 in group_list for g3 in group_list]            
    group = {'A': 'θ','C': 'φ','D': 'ξ','E': 'ξ','F': 'λ','G': 'θ','H': 'μ','I': 'θ','K': 'μ','L': 'θ',
             'M': 'θ','N': 'φ','P': 'φ','Q': 'φ','R': 'μ','S': 'φ','T': 'φ','V': 'θ','W': 'λ','Y': 'λ'}
    result = []
    for i in range(len(seq_list)):
        group_seq = [group.get(seq_list[i][n]) for n in range(len(seq_list[i]))]
        three_mers = Kmers_funct("".join(group_seq),3)
        m = [0] * len(group_comb)
        for j in range(len(group_comb)):
            if len(seq_list[i])-2 != 0:
                m[j] = three_mers.count(group_comb[j])/(len(seq_list[i])-2)
        result.append(m)
        
    return result

#Read the file, train the model and test on independent sets

in_file = sys.argv[1]  

train_seq, train_label = Prepare_data('..\\data_cashe\\ACP20mainTrain.fasta')
test_seq, test_label = Prepare_data('..\\data_cashe\\'+'ACP20main'+in_file+'.fasta')

train_X = np.c_[OE(train_seq),AAC(train_seq),DPC(train_seq),GTPC(train_seq)]
train_y = train_label
test_X = np.c_[OE(test_seq),AAC(test_seq),DPC(test_seq),GTPC(test_seq)]
test_y = test_label

LGBM = LGBMClassifier(n_estimators=50)
LGBM.fit(train_X, train_y)
resample_pred=LGBM.predict(np.array(test_X))
if in_file == 'New':
    print(LGBM.predict_proba(np.array(test_X))[:,1])
else:
    print('Sn:',"{:.4f}".format(metrics.recall_score(test_y,resample_pred)))
    print('Sp:',"{:.4f}".format(calculate_metric(test_y, resample_pred)))
    print('Acc:',"{:.4f}".format(metrics.accuracy_score(test_y,resample_pred)))