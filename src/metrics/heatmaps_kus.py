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
        # X, y, index = pickle.load(open(f'/mnt/d/Calvin/FYP/SMOTE/cicids2018_SMOTE_{level.value}.pkl','rb'))
        # print('loaded pickle')
        # _, _, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)  # shuffle and split
        # y_test.drop(index.intersection(y_test.index), inplace=True)
        # classes = np.unique(y_test.astype(str))
        # if level == Level.CATEGORY:
        #     label_decoding = {1: 'Bot', 2: 'DoS', 3: 'Brute Force', 4: 'Injection', 5: 'Infiltration', 0: 'Benign'}
        # else:
        #     label_decoding = {1: 'Bot2', 2: 'Bot1', 3: 'DoS-SlowHTTPTest', 4: 'DoS-Hulk', 5: 'Brute Force -Web2', 6: 'Brute Force -XSS2', 7: 'SQL Injection2', 8: 'Infiltration4', 9: 'Infiltration3', 10: 'DoS-GoldenEye', 11: 'DoS-Slowloris', 12: 'Brute Force -Web1', 13: 'Brute Force -XSS1', 14: 'SQL Injection1', 15: 'FTP-BruteForce', 16: 'SSH-BruteForce', 17: 'DDoS-HOIC', 18: 'DDoS-LOIC-UDP', 19: 'Infiltration1', 20: 'Infiltration2', 21: 'DDoS-LOIC-HTTP', 0: 'Benign'}
        for mode in Mode:
            for algo in ['svm', 'rf']:
                PATH = f'output/{algo}/'
                result_files = [folder+'/' for folder in os.listdir(PATH) if level.value in folder and (mode.value in folder or 'baseline' in folder)]
                acc: list[list] = []
                for folder in result_files:
                    folder2 = os.listdir(PATH+folder)[0]+'/'
                    file = os.listdir(PATH+folder+folder2)[0]
                    output_string = open(PATH+folder+folder2+file, 'r').read()
                    
                    lbl = int(re.findall(r'\d+', folder)[0])
                    recalls: list[int|str] = [int(lbl)]
                    for line in output_string.split(sep='\n'):
                        data = line.split()
                        if len(data) > 0 and data[0].replace('.','').isnumeric():
                            recalls.append(data[2])
                    acc.append(recalls)
                cols = ['idx', *range(len(recalls)-1)]
                
                dat = pd.DataFrame(acc, columns=cols)
                dat.sort_values(by='idx', inplace=True)
                dat.set_index('idx',inplace=True,drop=True)
                # dat.drop(columns=['sort'], inplace=True)
                # else:
                #     dat.sort_values(by='sort', inplace=True)
                #     dat.set_index('idx', inplace=True, drop=True)
                #     dat.drop(columns=['sort'], inplace=True)
                dat = dat.astype('float32')

                seaborn.heatmap(dat,annot=level==Level.CATEGORY, fmt='.2f')
                mode_s = 'Trained' if mode == Mode.INC else 'Omitted'
                level_s = 'Attacks' if level == Level.ATTACK else 'Attack Categories'
                plt.gca().set_xlabel(f'Classified {level_s}')
                plt.gca().set_ylabel(f'{mode_s} {level_s}')
                plt.title(algo.upper())
                plt.tight_layout()
                plt.savefig(f'output/heatmaps/{algo}_{mode.value}_{level.value}.pdf')
                plt.clf()
