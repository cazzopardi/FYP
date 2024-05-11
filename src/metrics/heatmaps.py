import os
import pickle
import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data.filtered_dataset import Level, Mode

if __name__ == '__main__':
    for level in Level:
        X, y, index = pickle.load(open(f'/mnt/d/Calvin/FYP/SMOTE/cicids2018_SMOTE_{level.value}.pkl','rb'))
        print('loaded pickle')
        _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
        y_test.drop(index.intersection(y_test.index), inplace=True)
        classes = np.unique(y_test.astype(str))
        if level == Level.CATEGORY:
            label_decoding = {1: 'Bot', 2: 'DoS', 3: 'Brute Force', 4: 'Injection', 5: 'Infiltration', 0: 'Benign'}
        else:
            label_decoding = {1: 'Bot2', 2: 'Bot1', 3: 'DoS-SlowHTTPTest', 4: 'DoS-Hulk', 5: 'Brute Force -Web2', 6: 'Brute Force -XSS2', 7: 'SQL Injection2', 8: 'Infiltration4', 9: 'Infiltration3', 10: 'DoS-GoldenEye', 11: 'DoS-Slowloris', 12: 'Brute Force -Web1', 13: 'Brute Force -XSS1', 14: 'SQL Injection1', 15: 'FTP-BruteForce', 16: 'SSH-BruteForce', 17: 'DDoS-HOIC', 18: 'DDoS-LOIC-UDP', 19: 'Infiltration1', 20: 'Infiltration2', 21: 'DDoS-LOIC-HTTP', 0: 'Benign'}
        for mode in Mode:
            result_files = os.listdir('results/supervised')
            rf_files = [file for file in result_files if ('random_forest' in file) and (str(level) in file) and (str(mode) in file or 'baseline' in file)]
            dt_files = [file for file in result_files if 'decision_tree' in file and (str(level) in file) and (str(mode) in file  or 'baseline' in file)]
            gb_files = [file for file in result_files if 'gradient_boosting' in file and (str(level) in file) and (str(mode) in file  or 'baseline' in file)]
            lda_files = [file for file in result_files if 'linear_discriminant_analysis' in file and (str(level) in file) and (str(mode) in file  or 'baseline' in file)]

            for algo, model_files in zip(['dt','rf','gb', 'lda'], [dt_files, rf_files, gb_files, lda_files]):
                acc: list[list] = []
                for file in model_files:
                    y_pred = pickle.load(open(f'results/supervised/{file}', 'rb'))
                    mask = np.logical_and(y_test != 0, y_pred)  # correct prediction mask
                    y_pred[mask] = y_test[mask].to_numpy()  # make prediction appear correct if the attack is flagged
                    
                    cf = classification_report(y_test.values, y_pred)
                    lbl = int(re.findall(r'\d+', file)[0])
                    recalls = [label_decoding[lbl], lbl]
                    for line in cf.split(sep='\n'):
                        data = line.split()
                        if len(data) > 0 and data[0] in classes:
                            recalls.append(data[2])
                    acc.append(recalls)
                if level == Level.CATEGORY:
                    cols = ['idx', 'sort', *[label_decoding[i] for i in range(len(recalls)-2)]]
                else:
                    cols = ['idx', 'sort', *[label_decoding[i] for i in range(14)], *[label_decoding[i] for i in range(15, len(recalls)-1)]]
                
                dat = pd.DataFrame(acc, columns=cols)
                if level == Level.CATEGORY:
                    dat.set_index('idx',inplace=True,drop=True)
                    dat.sort_values(by='sort', inplace=True)
                    dat.drop(columns=['sort'], inplace=True)
                else:
                    dat.sort_values(by='sort', inplace=True)
                    dat.set_index('idx', inplace=True, drop=True)
                    dat.drop(columns=['sort'], inplace=True)
                dat = dat.astype('float32')

                seaborn.heatmap(dat,annot=level==Level.CATEGORY)
                mode_s = 'Trained' if mode == Mode.INC else 'Omitted'
                level_s = 'Attacks' if level == Level.ATTACK else 'Attack Categories'
                plt.gca().set_xlabel(f'Classified {level_s}')
                plt.gca().set_ylabel(f'{mode_s} {level_s}')
                plt.tight_layout()
                plt.title(algo.upper())
                plt.savefig(f'results/heatmaps/{algo}_{mode.value}_{level.value}.pdf')
                plt.clf()
