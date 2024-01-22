import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from itertools import product
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix 

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

##############################################Ordinal encoding (50D)

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

############################################Amino acid composition (20D)

def AAC(seq_list):
    result = []
    for i in range(len(seq_list)):
        m = [0] * len(AA)
        for j in range(len(AA)):
            m[j] = seq_list[i].count(AA[j])/len(seq_list[i])
        result.append(m)
    return result

###########################################Dipeptide composition (210D)

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

########################################Grouped Amino Acids Encoding Composition (5D)

def GAEC(seq_list):
    oe = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
          'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    group_list = ['θ', 'λ', 'μ', 'ξ', 'φ']
    group = {'A': 'θ', 'C': 'φ', 'D': 'ξ', 'E': 'ξ', 'F': 'λ', 'G': 'θ', 'H': 'μ', 'I': 'θ', 'K': 'μ', 'L': 'θ',
             'M': 'θ', 'N': 'φ', 'P': 'φ', 'Q': 'φ', 'R': 'μ', 'S': 'φ', 'T': 'φ', 'V': 'θ', 'W': 'λ', 'Y': 'λ'}
    result = []
    for seq in seq_list:
        group_seq = [group.get(aa) for aa in seq]
        m = [0] * len(group_list)
        for j in range(len(group_list)):
            m[j] = sum(oe.get(seq[n]) * (group_seq[n] == group_list[j]) for n in range(len(seq))) / len(seq)
        result.append(m)
    return result

########################################Grouped Dipeptide  Encoding Composition (25D)

def GDPC(seq_list):
    oe = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
          'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    group_list = ['θ', 'λ', 'μ', 'ξ', 'φ']
    combined_group_list = [f"{a}{b}" for a, b in product(group_list, repeat=2)]
    group = {'A': 'θ', 'C': 'φ', 'D': 'ξ', 'E': 'ξ', 'F': 'λ', 'G': 'θ', 'H': 'μ', 'I': 'θ', 'K': 'μ', 'L': 'θ',
             'M': 'θ', 'N': 'φ', 'P': 'φ', 'Q': 'φ', 'R': 'μ', 'S': 'φ', 'T': 'φ', 'V': 'θ', 'W': 'λ', 'Y': 'λ'}
    result = []
    for seq in seq_list:
        two_mers_seq = Kmers_funct(seq,2)
        group_seq = [group.get(aa) for aa in seq]
        two_mers_group = Kmers_funct("".join(group_seq),2)
        m = [0] * len(combined_group_list)
        for j in range(len(combined_group_list)):
            m[j] = sum((oe.get(two_mers_seq[n][0])+oe.get(two_mers_seq[n][1])) * (two_mers_group[n] == combined_group_list[j]) for n in range(len(two_mers_seq))) / len(two_mers_seq)
        result.append(m)
    return result

########################################Grouped Tripeptide Encoding Composition (125D)

def GTEC(seq_list):
    oe = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
          'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    group_list = ['θ', 'λ', 'μ', 'ξ', 'φ']
    combined_group_list = [f"{a}{b}{c}" for a, b, c in product(group_list, repeat=3)]
    group = {'A': 'θ', 'C': 'φ', 'D': 'ξ', 'E': 'ξ', 'F': 'λ', 'G': 'θ', 'H': 'μ', 'I': 'θ', 'K': 'μ', 'L': 'θ',
             'M': 'θ', 'N': 'φ', 'P': 'φ', 'Q': 'φ', 'R': 'μ', 'S': 'φ', 'T': 'φ', 'V': 'θ', 'W': 'λ', 'Y': 'λ'}
    result = []
    for seq in seq_list:
        three_mers_seq = Kmers_funct(seq,3)
        group_seq = [group.get(aa) for aa in seq]
        three_mers_group = Kmers_funct("".join(group_seq),3)
        m = [0] * len(combined_group_list)
        for j in range(len(combined_group_list)):
            if len(seq)-2 != 0:
                m[j] = sum((oe.get(three_mers_seq[n][0])+oe.get(three_mers_seq[n][1])+oe.get(three_mers_seq[n][2])) * (three_mers_group[n] == combined_group_list[j]) for n in range(len(three_mers_seq))) / len(three_mers_seq)
        result.append(m)
    return result

#Read the file, train the model and test on independent sets

def calculate_specificity(y_true, y_pred): 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity
    
in_file = sys.argv[1]  

train_seq, train_y= Prepare_data('ACP20mainTrain.fasta')
test_seq, test_y = Prepare_data('ACP20main'+in_file+'.fasta')

train_X = np.c_[OE(train_seq),AAC(train_seq),DPC(train_seq),GTEC(train_seq)]
test_X = np.c_[OE(test_seq),AAC(test_seq),DPC(test_seq),GTEC(test_seq)]

LGBM = LGBMClassifier(n_estimators=100)
LGBM.fit(train_X, train_y)
resample_pred=LGBM.predict(np.array(test_X))
if in_file == 'New':
    print(LGBM.predict_proba(np.array(test_X))[:,1])
else:
    print('Sn:',"{:.4f}".format(metrics.recall_score(test_y,resample_pred)))
    print('Sp:',"{:.4f}".format(calculate_specificity(test_y, resample_pred)))
    print('MCC:',"{:.2f}".format(metrics.matthews_corrcoef(test_y,resample_pred)))
    print('Acc:',"{:.4f}".format(metrics.accuracy_score(test_y,resample_pred)))