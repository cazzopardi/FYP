
import os
import pickle
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from data.filtered_dataset import Level, Mode

if __name__ == '__main__':

    for level in Level:
        X, y, index = pickle.load(open(f'/mnt/d/Calvin/FYP/SMOTE/cicids2018_SMOTE_{level.value}.pkl','rb'))
        print('loaded pickle')
        _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
            
        y_test.drop(index.intersection(y_test.index), inplace=True)
        
        # replication
        # _, _, _, y_test = pickle.load(open('/mnt/d/Calvin/FYP/temp/smoted_replication.pkl', 'rb'))
        # result_files = os.listdir('results/replication')
        
        result_files = os.listdir('results/supervised')
        rf_files = [file for file in result_files if ('random_forest' in file) and (str(level) in file) and (str(Mode.EXC) in file)]
        dt_files = [file for file in result_files if 'decision_tree' in file and (str(level) in file) and (str(Mode.EXC) in file)]
        gb_files = [file for file in result_files if 'gradient_boosting' in file and (str(level) in file) and (str(Mode.EXC) in file)]
        lda_files = [file for file in result_files if 'linear_discriminant_analysis' in file and (str(level) in file) and (str(Mode.EXC) in file)]
        for model_files in [lda_files, dt_files, rf_files, gb_files]:
            accs: dict[str,float] = {lbl:0.0 for lbl in np.unique(y_test)}  # accuracy accumulator
            f1s: dict[str,float] = {lbl:0.0 for lbl in np.unique(y_test)}
            agg_acc = 0
            recall = 0
            prec = 0
            f1 = 0
            rec_unk = 0

            for file in model_files:
                y_pred = pickle.load(open(f'results/supervised/{file}', 'rb'))
                mask = np.logical_and(y_test != 0, y_pred)  # correct prediction mask
                y_pred[mask] = y_test[mask].to_numpy()  # make prediction appear correct if the attack is flagged
                
                cf = classification_report(y_test.to_numpy(), y_pred)
                # aggreagte acc
                for line in cf.split(sep='\n'):
                    data = line.split()
                    if len(data) > 0 and data[0] == 'accuracy':
                        agg_acc += float(data[-2])
                # aggreagate rec, prec and f1
                for line in cf.split(sep='\n'):
                    data = line.split()
                    if len(data) > 0 and data[0] == 'macro':
                        prec += float(data[2])
                        recall += float(data[3])
                        f1 += float(data[4])
                # recall-unk
                lbl = re.findall(r'\d+', file)[0]  # find excluded label
                for line in cf.split(sep='\n'):
                    data = line.split()
                    if len(data) > 0 and data[0] == lbl:
                        rec_unk += float(data[2])
                # per class acc
                for attack_label in np.unique(y_test):
                    mask = y_test == attack_label
                    acc = accuracy_score(y_test[mask], y_pred[mask])
                    accs[attack_label] += acc
                # per class f1
                for line in cf.split(sep='\n'):
                    data = line.split()
                    if len(data) > 0 and data[0] in str(np.unique(y_test)):
                        f1s[int(data[0])] += float(data[-2])
            accs = {k:v/len(model_files) for k,v in accs.items()}
            f1s = {k:v/len(model_files) for k,v in f1s.items()}
            print()
            print('Accuracy:', agg_acc/len(model_files))
            print('Recall:', recall/len(model_files))
            print('Recall-Unk', rec_unk/len(model_files))
            print('Precisions:', prec/len(model_files))
            print('F1:', f1/len(model_files))
            print('per class acc:', accs)
            print('Per class F1:', f1s)
